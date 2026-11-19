#!/usr/bin/env python3
"""Verify visual CUDA-event -> scheduler request -> OpenAI response joins."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSONL {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-jsonl", type=Path, required=True)
    parser.add_argument("--scheduler-jsonl", type=Path, required=True)
    parser.add_argument("--server-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    smoke_rows = read_jsonl(args.smoke_jsonl)
    scheduler_rows = read_jsonl(args.scheduler_jsonl)
    server_rows = read_jsonl(args.server_jsonl)
    image_response_rows = [
        row for row in smoke_rows
        if row.get("kind") == "image" and row.get("response_id") is not None
    ]
    response_rows = image_response_rows or [
        row for row in smoke_rows if row.get("response_id") is not None
    ]
    responses = {str(row["response_id"]): row for row in response_rows}
    scheduler_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in scheduler_rows:
        if row.get("kind") != "scheduler_step":
            continue
        encoder_inputs = row.get("scheduled_encoder_inputs", {})
        if isinstance(encoder_inputs, dict):
            for request_id, indices in encoder_inputs.items():
                if indices:
                    scheduler_by_id.setdefault(str(request_id), []).append(row)

    api_results = {
        str(row.get("request_id"))
        for row in server_rows
        if row.get("kind") == "engine_result_ready_to_api"
    }
    api_yields = {
        str(row.get("request_id"))
        for row in server_rows
        if row.get("kind") == "api_yield"
    }
    events: list[dict[str, Any]] = []
    flush_results: list[dict[str, Any]] = []
    for row in smoke_rows:
        if row.get("label") == "cuda-event-flush":
            flush_results.extend(row.get("flush_result", {}).get("results", []))
        elif row.get("kind") == "flush" and row.get("status") == 200:
            # Newer focused smoke clients store the collective_rpc response
            # directly as {kind: "flush", status: 200, body: {results: [...]}}.
            body = row.get("body", {})
            if isinstance(body, dict):
                flush_results.extend(body.get("results", []))
    for worker_idx, worker in enumerate(flush_results):
        for event in worker.get("cuda_events", []):
            is_visual = (
                event.get("mode") is not None
                and bool(event.get("mm_hashes"))
                and bool(event.get("request_ids"))
            )
            is_h2d = event.get("stage") == "mm_input_h2d"
            if not (is_visual or is_h2d):
                continue  # discard startup/profile visual event and unrelated stages
            event_request_ids = [str(x) for x in event.get("request_ids", [])]
            joined = []
            for scheduler_request_id in event_request_ids:
                scheduler_matches = scheduler_by_id.get(scheduler_request_id, [])
                response_matches = [
                    response_id
                    for response_id in responses
                    if scheduler_request_id.startswith(response_id + "-")
                ]
                joined.append(
                    {
                        "scheduler_request_id": scheduler_request_id,
                        "scheduler_encoder_rows": len(scheduler_matches),
                        "response_id_candidates": response_matches,
                        "api_result_present": bool(
                            len(response_matches) == 1
                            and response_matches[0] in api_results
                        ),
                        "api_yield_present": bool(
                            len(response_matches) == 1
                            and response_matches[0] in api_yields
                        ),
                    }
                )
            events.append(
                {
                    "worker_index": worker_idx,
                    "stage": event.get("stage", "visual_encoder"),
                    "mm_hash": event.get("mm_hash"),
                    "mm_hashes": event.get("mm_hashes", []),
                    "measurement": event.get("measurement"),
                    "cuda_elapsed_ms": event.get("cuda_elapsed_ms"),
                    "cuda_payload_bytes": event.get("cuda_payload_bytes"),
                    "tensor_shape": event.get("tensor_shape"),
                    "tensor_dtype": event.get("tensor_dtype"),
                    "source_is_pinned": event.get("source_is_pinned"),
                    "request_joins": joined,
                }
            )

    errors: list[str] = []
    if not responses:
        errors.append("no API response rows in smoke JSONL")
    visual_events = [event for event in events if event["stage"] == "visual_encoder"]
    h2d_events = [event for event in events if event["stage"] == "mm_input_h2d"]
    if not visual_events:
        errors.append("no request-scoped visual CUDA events in flush result")
    if not h2d_events:
        errors.append("no request-scoped multimodal H2D CUDA events in flush result")
    seen_visual_responses: set[str] = set()
    seen_h2d_responses: set[str] = set()
    for idx, event in enumerate(events):
        if not event["request_joins"]:
            errors.append(f"visual event {idx} has no mapped scheduler request ID")
        for join in event["request_joins"]:
            if join["scheduler_encoder_rows"] < 1:
                errors.append(
                    f"scheduler request {join['scheduler_request_id']} has no encoder row"
                )
            if len(join["response_id_candidates"]) != 1:
                errors.append(
                    f"scheduler request {join['scheduler_request_id']} did not map to one response"
                )
            elif not join["api_result_present"] or not join["api_yield_present"]:
                errors.append(
                    f"response {join['response_id_candidates'][0]} lacks engine/API trace rows"
                )
            else:
                if event["stage"] == "visual_encoder":
                    seen_visual_responses.add(join["response_id_candidates"][0])
                elif event["stage"] == "mm_input_h2d":
                    seen_h2d_responses.add(join["response_id_candidates"][0])
        if event["cuda_elapsed_ms"] is None:
            errors.append(f"visual event {idx} lacks CUDA elapsed time")

    missing_responses = sorted(set(responses) - seen_visual_responses)
    if missing_responses:
        errors.append(f"image response IDs without a visual event: {missing_responses}")
    missing_h2d = sorted(set(responses) - seen_h2d_responses)
    if missing_h2d:
        errors.append(f"image response IDs without an H2D copy event: {missing_h2d}")

    result = {
        "schema_version": 1,
        "valid": not errors,
        "source_files": {
            "smoke": str(args.smoke_jsonl),
            "scheduler": str(args.scheduler_jsonl),
            "server": str(args.server_jsonl),
        },
        "response_ids": sorted(responses),
        "visual_event_count": len(visual_events),
        "h2d_event_count": len(h2d_events),
        "responses_with_h2d_event": sorted(seen_h2d_responses),
        "h2d_payload_bytes": sum(
            int(event.get("cuda_payload_bytes", 0) or 0) for event in h2d_events
        ),
        "events": events,
        "errors": errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"valid": result["valid"], "responses": len(responses), "visual_events": len(visual_events), "h2d_events": len(h2d_events), "responses_with_h2d": len(seen_h2d_responses), "h2d_payload_bytes": result["h2d_payload_bytes"], "errors": errors}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
