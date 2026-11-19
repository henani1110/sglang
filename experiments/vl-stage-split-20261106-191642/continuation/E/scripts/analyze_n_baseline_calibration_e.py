#!/usr/bin/env python3
"""Audit descriptive outputs from an E/N calibration block; never labels it formal."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path, PurePosixPath
from statistics import mean


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    pos = (len(ordered) - 1) * probability
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def summary(values: list[float]) -> dict:
    return {
        "n": len(values),
        "p50": quantile(values, 0.50),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "max": max(values) if values else None,
        "mean": mean(values) if values else None,
    }


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(block: Path) -> dict:
    config = read_json(block / "block-config.json")
    rows = [json.loads(line) for line in (block / "run-index.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    # The remote output paths are authoritative in the run index; map them into this local mirror.
    records = []
    failures = []
    for index_row in rows:
        output_name = index_row.get("output_json")
        if not output_name:
            if index_row.get("status"):
                failures.append({"index_status": index_row.get("status"), "row": index_row})
            continue
        remote_path = PurePosixPath(output_name)
        # v0.1 and v0.2 share the same stable run layout; derive relative path after block ID.
        parts = remote_path.parts
        rel = PurePosixPath(*parts[parts.index(config["block_id"]) + 1 :])
        local_path = block.joinpath(*rel.parts)
        if not local_path.is_file():
            failures.append({"missing_output": str(local_path), "index_row": index_row})
            continue
        try:
            data = read_json(local_path)
        except Exception as exc:
            failures.append({"invalid_json": str(local_path), "error": repr(exc)})
            continue
        record = {"path": str(local_path), "index": index_row, "data": data}
        records.append(record)

    warmup = [item for item in records if item["index"].get("warmup_excluded")]
    sample = [item for item in records if not item["index"].get("warmup_excluded")]
    per_cell = {}
    all_http = []
    all_lane_counts = []
    for record in sample:
        ix, data = record["index"], record["data"]
        key = f"{ix.get('image_size')}px-{ix.get('text_lanes')}lane-{ix.get('arrival_pattern')}"
        cell = per_cell.setdefault(key, {
            "runs": 0, "seeds": [], "http_status_counts": {}, "lane_count_at_arrival": [],
            "image_ttft_p50_per_run_ms": [], "image_ttft_p95_per_run_ms": [],
            "image_ttft_max_per_run_ms": [], "text_gap_max_per_lane_ms": [],
            "text_finished_by_length_lanes": 0, "text_lanes": 0,
            "image_requests": 0, "image_finish_length": 0, "image_completion_tokens": [],
            "request_failures": [], "image_gate_failures": 0, "_lane_counts_raw": [],
        })
        cell["runs"] += 1
        cell["seeds"].append(ix.get("seed"))
        if data.get("image_gate_failure"):
            cell["image_gate_failures"] += 1
        for lane in data.get("text_results", []):
            status = str(lane.get("status"))
            cell["http_status_counts"][status] = cell["http_status_counts"].get(status, 0) + 1
            all_http.append(lane.get("status"))
            cell["text_lanes"] += 1
            if lane.get("finish_reason") == "length":
                cell["text_finished_by_length_lanes"] += 1
            gaps = lane.get("client_token_gaps_ms") or []
            if gaps:
                cell["text_gap_max_per_lane_ms"].append(max(gaps))
            if lane.get("status") != 200:
                cell["request_failures"].append({"kind": lane.get("kind"), "status": lane.get("status")})
        image_ttf = []
        for image in data.get("image_results", []):
            result = image.get("request_result", {})
            status = str(result.get("status"))
            cell["http_status_counts"][status] = cell["http_status_counts"].get(status, 0) + 1
            all_http.append(result.get("status"))
            cell["image_requests"] += 1
            if result.get("finish_reason") == "length":
                cell["image_finish_length"] += 1
            usage = result.get("usage") or {}
            if usage.get("completion_tokens") is not None:
                cell["image_completion_tokens"].append(usage["completion_tokens"])
            ttft = image.get("image_ttft_from_arrival_ms")
            if ttft is not None:
                image_ttf.append(float(ttft))
            if result.get("status") != 200:
                cell["request_failures"].append({"kind": result.get("kind"), "status": result.get("status"), "error": result.get("error_body")})
            decision = image.get("controller_decision") or {}
            lane_count = decision.get("text_lane_count")
            if lane_count is not None:
                cell["lane_count_at_arrival"].append(int(lane_count))
                cell["_lane_counts_raw"].append(int(lane_count))
                all_lane_counts.append(int(lane_count))
        if image_ttf:
            cell["image_ttft_p50_per_run_ms"].append(quantile(image_ttf, 0.50))
            cell["image_ttft_p95_per_run_ms"].append(quantile(image_ttf, 0.95))
            cell["image_ttft_max_per_run_ms"].append(max(image_ttf))

    for cell in per_cell.values():
        for key in ("lane_count_at_arrival", "image_ttft_p50_per_run_ms", "image_ttft_p95_per_run_ms",
                    "image_ttft_max_per_run_ms", "text_gap_max_per_lane_ms", "image_completion_tokens"):
            cell[key] = summary(cell[key])
        raw_counts = cell.pop("_lane_counts_raw")
        target_lanes = int(next(k for k, v in per_cell.items() if v is cell).split("lane-")[0].split("px-")[-1])
        cell["observed_lane_count_min"] = min(raw_counts) if raw_counts else None
        cell["observed_lane_count_max"] = max(raw_counts) if raw_counts else None
        cell["all_target_lanes_active_at_all_arrivals"] = (
            cell["lane_count_at_arrival"]["n"] == cell["runs"] * config["image_count_per_cell"]
            and raw_counts
            and all(count == target_lanes for count in raw_counts)
        )
    return {
        "block_id": config["block_id"],
        "block_label": config.get("block_label"),
        "formal_run": False,
        "arm": config.get("arm"),
        "config": config,
        "index_rows": len(rows),
        "loaded_outputs": len(records),
        "warmup_outputs": len(warmup),
        "measurement_outputs": len(sample),
        "http_status_counts": {str(status): all_http.count(status) for status in sorted(set(all_http), key=lambda x: str(x))},
        "all_arrival_lane_count_distribution": {str(count): all_lane_counts.count(count) for count in sorted(set(all_lane_counts))},
        "cells": per_cell,
        "failures": failures,
        "formal_run_count_added": 0,
        "limitations": [
            "Synthetic engineering calibration only; no answer-quality, pruning, or treatment comparison.",
            "The v0.1 uniform interval is 750 ms, while 512-token text streams completed in a few seconds; later image arrivals can therefore lack active text lanes.",
            "Descriptive per-run summaries are not independent-token inference and do not establish SLO goodput.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("block", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.block)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("block_id", "index_rows", "loaded_outputs", "warmup_outputs", "measurement_outputs", "http_status_counts", "all_arrival_lane_count_distribution", "formal_run_count_added", "failures")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
