#!/usr/bin/env python3
"""Join concurrent per-request image-budget smoke output to vLLM traces."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise SystemExit(f"expected JSON object at {path}:{line_no}")
            rows.append(value)
    return rows


def matches_request(request_id: str, response_id: str) -> bool:
    return request_id == response_id or request_id.startswith(response_id + "-")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-jsonl", type=Path, required=True)
    parser.add_argument("--server-jsonl", type=Path, required=True)
    parser.add_argument("--scheduler-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--full-image-tokens", type=int, default=576)
    parser.add_argument("--pixel-rows", type=int, default=2304)
    parser.add_argument("--prompt-overhead", type=int, default=21)
    parser.add_argument("--kv-block-size", type=int, default=16)
    args = parser.parse_args()

    client_rows = read_jsonl(args.client_jsonl)
    server_rows = read_jsonl(args.server_jsonl)
    scheduler_rows = read_jsonl(args.scheduler_jsonl)
    flush_rows = [row for row in client_rows if row.get("label") == "cuda-event-flush"]
    requests = [row for row in client_rows if row.get("keep_ratio") is not None]
    errors: list[str] = []
    outcomes: list[dict[str, Any]] = []

    for row in requests:
        label = str(row.get("label"))
        ratio = float(row["keep_ratio"])
        expected_tokens = min(
            args.full_image_tokens,
            max(1, math.ceil(args.full_image_tokens * ratio)),
        )
        response_id = str(row.get("response_id") or "")
        matching_visuals = [
            event
            for event in server_rows
            if event.get("kind") == "visual_forward"
            and any(
                matches_request(str(request_id), response_id)
                for request_id in event.get("request_ids", [])
            )
        ]
        matching_scheduler: list[dict[str, Any]] = []
        for step in scheduler_rows:
            for scheduled in step.get("scheduled_new_reqs", []):
                request_id = str(scheduled.get("request_id", ""))
                if matches_request(request_id, response_id):
                    matching_scheduler.append(
                        {
                            "request_id": request_id,
                            "prompt_len": scheduled.get("prompt_len"),
                            "num_computed_tokens": scheduled.get("num_computed_tokens"),
                            "block_ids_by_group": scheduled.get("block_ids_by_group"),
                        }
                    )

        usage = row.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        visual_shapes = [event.get("output_shape") for event in matching_visuals]
        pixel_shapes = [event.get("pixel_shape") for event in matching_visuals]
        actual_token_counts = [
            int(shape[0])
            for shape in visual_shapes
            if isinstance(shape, list) and shape
        ]
        scheduled_prompt_lens = [
            int(item["prompt_len"])
            for item in matching_scheduler
            if item.get("prompt_len") is not None
        ]
        kv_blocks = [
            sum(len(group) for group in item.get("block_ids_by_group", []))
            for item in matching_scheduler
        ]

        outcome = {
            "label": label,
            "keep_ratio": ratio,
            "expected_image_keep_count": expected_tokens,
            "status": row.get("status"),
            "response_id": response_id or None,
            "prompt_tokens": prompt_tokens,
            "prompt_overhead": (
                int(prompt_tokens) - expected_tokens
                if prompt_tokens is not None
                else None
            ),
            "visual_output_shapes": visual_shapes,
            "visual_pixel_shapes": pixel_shapes,
            "actual_visual_token_counts": actual_token_counts,
            "scheduler_initial_prompt_lens": scheduled_prompt_lens,
            "scheduler_kv_block_counts": kv_blocks,
            "visual_event_count": len(matching_visuals),
            "scheduler_new_request_rows": len(matching_scheduler),
        }
        outcomes.append(outcome)

        if row.get("status") != 200:
            errors.append(f"{label}: request status is {row.get('status')}")
        if not response_id:
            errors.append(f"{label}: response id is missing")
        if actual_token_counts != [expected_tokens]:
            errors.append(
                f"{label}: expected one visual output with {expected_tokens} rows, got {actual_token_counts}"
            )
        if pixel_shapes != [[args.pixel_rows, 1536]]:
            errors.append(f"{label}: unexpected ViT input shapes {pixel_shapes}")
        expected_prompt = args.prompt_overhead + expected_tokens
        if prompt_tokens != expected_prompt:
            errors.append(
                f"{label}: expected API prompt usage {expected_prompt}, got {prompt_tokens}"
            )
        if scheduled_prompt_lens != [expected_prompt]:
            errors.append(
                f"{label}: scheduler prompt lengths do not match {expected_prompt}: {scheduled_prompt_lens}"
            )
        expected_blocks = math.ceil(expected_prompt / args.kv_block_size)
        if kv_blocks != [expected_blocks]:
            errors.append(
                f"{label}: expected {expected_blocks} KV blocks, got {kv_blocks}"
            )

    ratios = sorted(row.get("keep_ratio") for row in requests)
    if ratios != [0.25, 0.75]:
        errors.append(f"expected the 0.25 and 0.75 request pair, got {ratios}")
    if not flush_rows or any(row.get("status") != 200 for row in flush_rows):
        errors.append("CUDA event flush is missing or did not return HTTP 200")

    result = {
        "schema_version": 1,
        "valid": not errors,
        "input_assumptions": {
            "full_image_tokens": args.full_image_tokens,
            "pixel_rows": args.pixel_rows,
            "prompt_overhead": args.prompt_overhead,
            "kv_block_size": args.kv_block_size,
        },
        "request_count": len(requests),
        "requests": outcomes,
        "cuda_flush_statuses": [row.get("status") for row in flush_rows],
        "errors": errors,
        "limitation": "Synthetic mechanism smoke only; not a quality, latency, or SLO result.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "valid": result["valid"],
                "request_count": len(requests),
                "actual_visual_token_counts": [
                    outcome["actual_visual_token_counts"] for outcome in outcomes
                ],
                "prompt_tokens": [outcome["prompt_tokens"] for outcome in outcomes],
                "errors": errors,
            },
            ensure_ascii=False,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
