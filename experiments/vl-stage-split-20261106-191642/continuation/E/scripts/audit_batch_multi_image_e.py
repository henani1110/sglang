#!/usr/bin/env python3
"""Audit co-scheduled images and multi-image request event attribution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise SystemExit(f"expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def response_candidates(request_id: str, response_ids: list[str]) -> list[str]:
    return [response_id for response_id in response_ids if request_id.startswith(response_id + "-")]


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
    response_records = {
        str(row["response_id"]): row
        for row in smoke_rows
        if row.get("response_id") is not None
    }
    response_ids = sorted(response_records)
    multi_image_responses = {
        response_id: row
        for response_id, row in response_records.items()
        if len(row.get("image_names", [])) > 1
    }

    co_scheduled_batches: list[dict[str, Any]] = []
    for row_idx, row in enumerate(scheduler_rows):
        encoder_inputs = row.get("scheduled_encoder_inputs", {})
        if not isinstance(encoder_inputs, dict) or not encoder_inputs:
            continue
        request_ids = [str(request_id) for request_id in encoder_inputs]
        mapped = {
            request_id: response_candidates(request_id, response_ids)
            for request_id in request_ids
        }
        response_batch = sorted(
            {
                candidates[0]
                for candidates in mapped.values()
                if len(candidates) == 1
            }
        )
        if len(response_batch) > 1:
            co_scheduled_batches.append(
                {
                    "scheduler_row_index": row_idx,
                    "request_ids": request_ids,
                    "response_ids": response_batch,
                    "scheduled_encoder_inputs": encoder_inputs,
                }
            )

    flush_rows = [row for row in smoke_rows if row.get("label") == "cuda-event-flush"]
    events: list[dict[str, Any]] = []
    for flush in flush_rows:
        for worker_idx, worker in enumerate(flush.get("flush_result", {}).get("results", [])):
            for event in worker.get("cuda_events", []):
                if event.get("stage") != "mm_input_h2d" and not (
                    event.get("mode") is not None
                    and event.get("mm_hashes")
                    and event.get("request_ids")
                ):
                    continue
                req_ids = [str(request_id) for request_id in event.get("request_ids", [])]
                mapped = {
                    request_id: response_candidates(request_id, response_ids)
                    for request_id in req_ids
                }
                events.append(
                    {
                        "worker_index": worker_idx,
                        "stage": event.get("stage", "visual_encoder"),
                        "mm_hashes": event.get("mm_hashes", []),
                        "request_ids": req_ids,
                        "response_ids": sorted(
                            {
                                candidates[0]
                                for candidates in mapped.values()
                                if len(candidates) == 1
                            }
                        ),
                        "cuda_elapsed_ms": event.get("cuda_elapsed_ms"),
                        "cuda_payload_bytes": event.get("cuda_payload_bytes"),
                    }
                )

    multi_image_checks: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for response_id, row in multi_image_responses.items():
        h2d = [
            event for event in events
            if event["stage"] == "mm_input_h2d" and response_id in event["response_ids"]
        ]
        visual = [
            event for event in events
            if event["stage"] == "visual_encoder" and response_id in event["response_ids"]
        ]
        multi_hash_h2d = [event for event in h2d if len(event["mm_hashes"]) >= 2]
        multi_hash_visual = [event for event in visual if len(event["mm_hashes"]) >= 2]
        check = {
            "label": row.get("label"),
            "image_names": row.get("image_names", []),
            "response_id": response_id,
            "h2d_event_count": len(h2d),
            "visual_event_count": len(visual),
            "multi_hash_h2d_event_count": len(multi_hash_h2d),
            "multi_hash_visual_event_count": len(multi_hash_visual),
            "h2d_events": h2d,
            "visual_events": visual,
        }
        multi_image_checks[response_id] = check
        if not h2d:
            errors.append(f"multi-image response {response_id} has no H2D event")
        if not visual:
            errors.append(f"multi-image response {response_id} has no visual event")
        if not multi_hash_h2d:
            errors.append(f"multi-image response {response_id} has no H2D event carrying >=2 mm_hashes")
        if not multi_hash_visual:
            errors.append(f"multi-image response {response_id} has no visual event carrying >=2 mm_hashes")

    server_response_ids = {
        str(row.get("response_id"))
        for row in server_rows
        if row.get("response_id") is not None
    }
    result = {
        "schema_version": 1,
        "valid": not errors,
        "response_ids": response_ids,
        "response_count": len(response_ids),
        "co_scheduled_image_batches": co_scheduled_batches,
        "co_scheduled_image_batch_count": len(co_scheduled_batches),
        "multi_image_requests": multi_image_checks,
        "multi_image_request_count": len(multi_image_checks),
        "events": events,
        "server_rows_with_response_id": sorted(server_response_ids),
        "errors": errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "valid": result["valid"],
                "responses": len(response_ids),
                "co_scheduled_image_batches": len(co_scheduled_batches),
                "multi_image_requests": len(multi_image_checks),
                "errors": errors,
            },
            ensure_ascii=False,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
