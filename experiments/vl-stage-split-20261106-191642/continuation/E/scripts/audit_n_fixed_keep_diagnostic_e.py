#!/usr/bin/env python3
"""Audit actual 0.5-keep visual rows, prompt accounting, and trace joins."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
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
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def matches_request(request_id: str, response_id: str) -> bool:
    return request_id == response_id or request_id.startswith(response_id + "-")


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)

    def percentile(p: float) -> float:
        position = (len(ordered) - 1) * p
        lo = math.floor(position)
        hi = math.ceil(position)
        if lo == hi:
            return ordered[lo]
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)

    return {"count": len(values), "p50": percentile(0.5), "p95": percentile(0.95), "max": max(values)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-index", type=Path, required=True)
    parser.add_argument("--server-jsonl", type=Path, required=True)
    parser.add_argument("--scheduler-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-keep-ratio", type=float, default=0.5)
    parser.add_argument("--kv-block-size", type=int, default=16)
    args = parser.parse_args()

    index_rows = read_jsonl(args.run_index)
    seed_rows = [row for row in index_rows if row.get("warmup_excluded") is False]
    server_rows = read_jsonl(args.server_jsonl)
    scheduler_rows = read_jsonl(args.scheduler_jsonl)
    visual_events = [row for row in server_rows if row.get("kind") == "visual_forward"]
    image_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    reports_by_path: dict[Path, dict[str, Any]] = {}
    expected_by_response: dict[str, dict[str, int]] = {}
    cell_counts: Counter[str] = Counter()
    ratio_counts: Counter[float] = Counter()
    image_ttft_p95_by_run: defaultdict[tuple[int, int, str], list[float]] = defaultdict(list)
    text_gap_p95_by_lane: defaultdict[tuple[int, int, str], list[float]] = defaultdict(list)

    for run in seed_rows:
        output_path = Path(run["output_json"])
        if not output_path.is_file():
            errors.append(f"missing replay JSON: {output_path}")
            continue
        report = json.loads(output_path.read_text(encoding="utf-8"))
        reports_by_path[output_path] = report
        grid = report.get("processor_grid_thw")
        if not isinstance(grid, list) or len(grid) != 3:
            continue
        full_tokens = int(report["full_decoder_tokens"])
        patch_rows = int(grid[0]) * int(grid[1]) * int(grid[2])
        expected_keep = max(1, math.ceil(full_tokens * args.expected_keep_ratio))
        for image in report.get("image_results", []):
            response_id = str((image.get("request_result") or {}).get("response_id") or "")
            if response_id:
                expected_by_response[response_id] = {
                    "keep_tokens": expected_keep,
                    "patch_rows": patch_rows,
                }

    for run in seed_rows:
        output_path = Path(run["output_json"])
        if not output_path.is_file():
            errors.append(f"missing replay JSON: {output_path}")
            continue
        report = reports_by_path.get(output_path)
        if report is None:
            continue
        cell = (int(run["image_size"]), int(run["text_lanes"]), str(run["arrival_pattern"]))
        cell_key = f"{cell[0]}-{cell[1]}lane-{cell[2]}"
        cell_counts[cell_key] += 1
        if run.get("returncode") != 0:
            errors.append(f"seed {run.get('seed')} {cell_key}: replay returncode={run.get('returncode')}")
        if not run.get("all_target_text_lanes_active_at_every_image_arrival"):
            errors.append(f"seed {run.get('seed')} {cell_key}: target lanes not active for every image arrival")
        if report.get("formal_run") is not False:
            errors.append(f"seed {run.get('seed')} {cell_key}: report is not marked non-formal")
        ratio = float(report.get("diagnostic_fixed_keep_ratio", -1))
        if not math.isclose(ratio, args.expected_keep_ratio, rel_tol=0, abs_tol=1e-12):
            errors.append(f"seed {run.get('seed')} {cell_key}: unexpected diagnostic ratio {ratio}")
        ratio_counts[ratio] += len(report.get("image_results", []))

        grid = report.get("processor_grid_thw")
        if not isinstance(grid, list) or len(grid) != 3:
            errors.append(f"seed {run.get('seed')} {cell_key}: invalid processor grid {grid}")
            continue
        full_tokens = int(report["full_decoder_tokens"])
        full_patch_rows = int(grid[0]) * int(grid[1]) * int(grid[2])
        expected_keep = max(1, math.ceil(full_tokens * args.expected_keep_ratio))
        ttft = report.get("image_ttft_from_arrival_summary", {}).get("p95_ms")
        if ttft is not None:
            image_ttft_p95_by_run[cell].append(float(ttft))
        for text_result in report.get("text_results", []):
            gaps = text_result.get("client_token_gaps_ms", [])
            if gaps:
                gap_summary = sorted(float(value) for value in gaps)
                index = (len(gap_summary) - 1) * 0.95
                low, high = math.floor(index), math.ceil(index)
                p95 = gap_summary[low] if low == high else gap_summary[low] + (gap_summary[high] - gap_summary[low]) * (index - low)
                text_gap_p95_by_lane[cell].append(p95)
            if text_result.get("status") != 200:
                errors.append(f"seed {run.get('seed')} {cell_key}: text lane status={text_result.get('status')}")

        for image in report.get("image_results", []):
            result = image.get("request_result", {})
            response_id = str(result.get("response_id") or "")
            status = result.get("status")
            if status != 200:
                errors.append(f"seed {run.get('seed')} {cell_key} {image.get('label')}: HTTP {status}")
            if not response_id:
                errors.append(f"seed {run.get('seed')} {cell_key} {image.get('label')}: missing response id")
            if image.get("controller_decision", {}).get("diagnostic_expected_keep_count") != expected_keep:
                errors.append(f"seed {run.get('seed')} {cell_key} {image.get('label')}: request expected keep count mismatch")
            if result.get("usage") is None or result["usage"].get("prompt_tokens") is None:
                errors.append(f"seed {run.get('seed')} {cell_key} {image.get('label')}: missing usage.prompt_tokens")

            matches = []
            for event in visual_events:
                request_ids = [str(value) for value in event.get("request_ids", [])]
                if any(matches_request(request_id, response_id) for request_id in request_ids):
                    matches.append(event)
            if len(matches) != 1:
                errors.append(f"seed {run.get('seed')} {cell_key} {image.get('label')}: expected one visual event membership, got {len(matches)}")
            actual_tokens = None
            pixel_rows = None
            if len(matches) == 1:
                event = matches[0]
                output_shape = event.get("output_shape")
                pixel_shape = event.get("pixel_shape")
                member_ids = [str(value) for value in event.get("request_ids", [])]
                members = [
                    response for response in expected_by_response
                    if any(matches_request(request_id, response) for request_id in member_ids)
                ]
                # The event may be a multi-request encoder batch. Audit the batch
                # output against the sum of all member requests' expected rows.
                expected_batch_tokens = sum(expected_by_response[response]["keep_tokens"] for response in members)
                expected_batch_patch_rows = sum(expected_by_response[response]["patch_rows"] for response in members)
                if isinstance(output_shape, list) and output_shape:
                    actual_tokens = int(output_shape[0])
                    if actual_tokens != expected_batch_tokens:
                        errors.append(f"seed {run.get('seed')} {cell_key} {image.get('label')}: visual batch rows {actual_tokens}, expected {expected_batch_tokens} for {len(members)} requests")
                if isinstance(pixel_shape, list) and pixel_shape:
                    pixel_rows = int(pixel_shape[0])
                    if pixel_rows != expected_batch_patch_rows:
                        errors.append(f"seed {run.get('seed')} {cell_key} {image.get('label')}: ViT batch rows {pixel_rows}, expected {expected_batch_patch_rows} for {len(members)} requests")

            usage = result.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            scheduler_matches = []
            for step in scheduler_rows:
                for scheduled in step.get("scheduled_new_reqs", []):
                    request_id = str(scheduled.get("request_id", ""))
                    if response_id and matches_request(request_id, response_id):
                        scheduler_matches.append(scheduled)
            prompt_lens = [int(row["prompt_len"]) for row in scheduler_matches if row.get("prompt_len") is not None]
            if len(prompt_lens) != 1:
                errors.append(f"seed {run.get('seed')} {cell_key} {image.get('label')}: expected one scheduler prompt row, got {prompt_lens}")
            elif prompt_tokens is not None and prompt_lens[0] != int(prompt_tokens):
                errors.append(f"seed {run.get('seed')} {cell_key} {image.get('label')}: API prompt={prompt_tokens}, scheduler prompt={prompt_lens[0]}")
            kv_blocks = [
                sum(len(group) for group in row.get("block_ids_by_group", []))
                for row in scheduler_matches
                if isinstance(row.get("block_ids_by_group"), list)
            ]

            image_rows.append({
                "seed": run.get("seed"),
                "image_size": cell[0],
                "text_lanes": cell[1],
                "arrival_pattern": cell[2],
                "label": image.get("label"),
                "response_id": response_id or None,
                "status": status,
                "full_decoder_tokens": full_tokens,
                "expected_keep_tokens_per_image": expected_keep,
                "actual_visual_event_output_rows": actual_tokens,
                "full_vit_patch_rows_expected": full_patch_rows,
                "visual_event_input_rows": pixel_rows,
                "prompt_tokens_api": prompt_tokens,
                "prompt_tokens_scheduler": prompt_lens,
                "scheduler_kv_block_counts": kv_blocks,
                "visual_event_count": len(matches),
                "scheduler_new_request_rows": len(scheduler_matches),
                "ttft_ms": image.get("image_ttft_from_arrival_ms"),
            })

    flush_statuses = [
        row.get("event_flush", {}).get("status")
        for run in seed_rows
        if Path(run["output_json"]) in reports_by_path
        for row in [reports_by_path[Path(run["output_json"])]]
    ]
    if any(status != 200 for status in flush_statuses):
        errors.append(f"one or more event flush calls failed: {Counter(flush_statuses)}")

    result = {
        "schema_version": 1,
        "experiment": "N fixed keep-ratio engineering diagnostic",
        "formal_run": False,
        "expected_keep_ratio": args.expected_keep_ratio,
        "planned_seed_cells": 96,
        "completed_seed_cells": len(seed_rows),
        "completed_cells_by_cell": dict(sorted(cell_counts.items())),
        "image_request_count": len(image_rows),
        "image_statuses": dict(Counter(str(row["status"]) for row in image_rows)),
        "ratio_image_request_counts": {str(key): value for key, value in ratio_counts.items()},
        "image_ttft_run_p95_ms_by_cell": {
            f"{key[0]}-{key[1]}lane-{key[2]}": summarize(values)
            for key, values in sorted(image_ttft_p95_by_run.items())
        },
        "text_gap_run_p95_ms_by_cell": {
            f"{key[0]}-{key[1]}lane-{key[2]}": summarize(values)
            for key, values in sorted(text_gap_p95_by_lane.items())
        },
        "event_flush_statuses": dict(Counter(str(value) for value in flush_statuses)),
        "images": image_rows,
        "errors": errors,
        "valid": not errors,
        "limitations": [
            "Synthetic chart images; fixed 0.5 ratio is a mechanism diagnostic, not a calibrated safe budget.",
            "Client-visible SSE and TTFT summaries are descriptive and do not establish strategy benefit or formal SLO performance.",
            "No answer-quality claim, rights-cleared benchmark result, or 480-run result is implied.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "valid": result["valid"],
        "completed_seed_cells": result["completed_seed_cells"],
        "image_request_count": result["image_request_count"],
        "image_statuses": result["image_statuses"],
        "errors": errors[:20],
        "error_count": len(errors),
    }, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
