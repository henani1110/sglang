#!/usr/bin/env python3
"""Replay fixed text/image arrivals through the live E admission controller."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from e_content_budget_policy import PreencoderBudgetConfig, raw_pixel_density_proxy
from e_live_joint_controller_e import TextDeadlineTracker, prepare_image_request_budget
from smoke_batch_multi_image_e import make_chart
from smoke_stage_split_e import flush_events, stream_request


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(np.max(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8285")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--text-lanes", type=int, choices=(2, 6), default=2)
    parser.add_argument("--image-size", type=int, choices=(512, 768), default=768)
    parser.add_argument("--text-max-tokens", type=int, default=256)
    parser.add_argument("--image-count", type=int, default=4)
    parser.add_argument("--arrival-pattern", choices=("uniform", "burst"), default="uniform")
    parser.add_argument("--arrival-interval-ms", type=float)
    parser.add_argument("--image-max-tokens", type=int, default=16)
    parser.add_argument("--first-token-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=20261109,
                        help="Deterministic paired trace seed; identical seeds produce identical chart assets and request seeds.")
    args = parser.parse_args()
    if args.text_max_tokens < 1 or args.image_count < 1 or args.image_max_tokens < 1:
        raise SystemExit("token limits and image count must be positive")
    if args.arrival_interval_ms is None:
        interval_ms = 750.0 if args.arrival_pattern == "uniform" else 25.0
    else:
        interval_ms = args.arrival_interval_ms
    if interval_ms < 0:
        raise SystemExit("arrival interval must be non-negative")

    asset_dir = Path(args.asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    images: list[tuple[str, bytes, str]] = []
    for index in range(args.image_count):
        label = f"R{index + 1}"
        color_seed = hashlib.sha256(f"{args.seed}:{index}".encode("utf-8")).digest()
        color = tuple(30 + value % 190 for value in color_seed[:3])
        payload = make_chart(label, (args.image_size, args.image_size), color)
        digest = hashlib.sha256(payload).hexdigest()
        asset = asset_dir / f"joint-replay-s{args.seed}-{args.image_size}-{args.arrival_pattern}-{label.lower()}.png"
        asset.write_bytes(payload)
        images.append((label, payload, digest))

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    first_image = Image.open(
        asset_dir / f"joint-replay-s{args.seed}-{args.image_size}-{args.arrival_pattern}-r1.png"
    ).convert("RGB")
    processed = processor.image_processor(images=[first_image], return_tensors="pt")
    grid = [int(value) for value in processed["image_grid_thw"][0].tolist()]
    merge_size = int(getattr(processor.image_processor, "merge_size", 2))
    full_tokens = grid[0] * grid[1] * grid[2] // (merge_size * merge_size)
    rgb = torch.from_numpy(np.asarray(first_image).copy()).permute(2, 0, 1).float() / 255.0
    pixel_score = float(raw_pixel_density_proxy(rgb))
    query = "Read the trend and chart title in one sentence."

    tracker = TextDeadlineTracker()
    first_content = [Event() for _ in range(args.text_lanes)]
    test_start_ns = time.monotonic_ns()
    executor = ThreadPoolExecutor(max_workers=args.text_lanes + args.image_count)
    text_jobs: list[Future] = []
    image_jobs: list[dict[str, Any]] = []
    gate_failure = None
    try:
        for lane in range(args.text_lanes):
            label = f"text-{lane + 1}"
            body = {
                "model": "qwen3vlpilot",
                "prompt": (
                    "Continue a numbered technical design checklist with short distinct items. "
                    "Keep generating until the token limit. 1. Keep scheduler decisions observable."
                ),
                "temperature": 0,
                "max_tokens": args.text_max_tokens,
                "seed": args.seed * 100 + lane,
                "stream": True,
            }
            text_jobs.append(
                executor.submit(
                    stream_request,
                    args.base_url.rstrip("/") + "/v1/completions",
                    label,
                    body,
                    first_content[lane],
                    test_start_ns,
                    lambda lane_id, started: tracker.record_token(lane_id, started),
                    lambda lane_id, token_ns: tracker.record_token(lane_id, token_ns),
                    lambda lane_id: tracker.finish_lane(lane_id),
                )
            )

        gate_deadline = time.monotonic() + args.first_token_timeout_seconds
        for index, event in enumerate(first_content):
            if not event.wait(max(0.0, gate_deadline - time.monotonic())):
                gate_failure = {
                    "error": "not all text lanes emitted first content before image replay",
                    "waiting_lane": index + 1,
                    "missing_lanes": [i + 1 for i, item in enumerate(first_content) if not item.is_set()],
                }
                break

        schedule_start_ns = time.monotonic_ns()
        image_endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
        for index, (label, payload, digest) in enumerate(images):
            scheduled_ns = schedule_start_ns + int(index * interval_ms * 1_000_000)
            remaining_ns = scheduled_ns - time.monotonic_ns()
            if remaining_ns > 0:
                time.sleep(remaining_ns / 1_000_000_000.0)
            arrival_ns = time.monotonic_ns()
            decision_ns = time.monotonic_ns()
            decision = prepare_image_request_budget(
                query=query,
                pixel_density_score=pixel_score,
                full_token_count=full_tokens,
                image_arrival_ns=arrival_ns,
                now_ns=decision_ns,
                tracker=tracker,
                text_max_gap_ms=30.0,
                image_ttft_ms=55.0,
                policy_config=PreencoderBudgetConfig(),
            )
            data_url = "data:image/png;base64," + __import__("base64").b64encode(payload).decode("ascii")
            body = {
                "model": "qwen3vlpilot",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]}],
                "temperature": 0,
                "max_tokens": args.image_max_tokens,
                "seed": args.seed * 1000 + index,
                "stream": True,
                "stream_options": {"include_usage": True},
                "mm_processor_kwargs": decision["mm_processor_kwargs"],
            }
            image_future = executor.submit(
                stream_request,
                image_endpoint,
                f"image-{index + 1}",
                body,
                None,
                test_start_ns,
                request_arrival_ns=arrival_ns,
            )
            image_jobs.append({
                "label": label,
                "image_sha256": digest,
                "request_seed": args.seed * 1000 + index,
                "scheduled_offset_ms": (scheduled_ns - schedule_start_ns) / 1e6,
                "actual_arrival_offset_ms": (arrival_ns - test_start_ns) / 1e6,
                "arrival_ns": arrival_ns,
                "controller_decision": decision,
                "future": image_future,
            })
    finally:
        executor.shutdown(wait=True, cancel_futures=False)

    text_results = [future.result() for future in text_jobs]
    image_results = []
    for job in image_jobs:
        result = job["future"].result()
        job_without_future = {key: value for key, value in job.items() if key not in {"future", "arrival_ns"}}
        image_results.append({
            **job_without_future,
            "request_result": result,
            "image_ttft_from_arrival_ms": result["client_token_times_ms"][0]
            if result.get("client_token_times_ms") else None,
        })

    text_gap_summaries = {
        item["kind"]: summarize(item["client_token_gaps_ms"])
        for item in text_results
    }
    image_ttft_values = [
        float(item["image_ttft_from_arrival_ms"])
        for item in image_results
        if item["image_ttft_from_arrival_ms"] is not None
    ]
    report = {
        "experiment": "fixed_arrival_live_joint_controller_replay",
        "formal_run": False,
        "seed": args.seed,
        "text_request_seeds": [args.seed * 100 + lane for lane in range(args.text_lanes)],
        "image_request_seeds": [args.seed * 1000 + index for index in range(args.image_count)],
        "arrival_pattern": args.arrival_pattern,
        "arrival_interval_ms": interval_ms,
        "text_lanes": args.text_lanes,
        "image_size": args.image_size,
        "text_max_tokens": args.text_max_tokens,
        "image_count": args.image_count,
        "image_max_tokens": args.image_max_tokens,
        "image_gate_failure": gate_failure,
        "processor_grid_thw": grid,
        "merge_size": merge_size,
        "full_decoder_tokens": full_tokens,
        "pixel_density_score": pixel_score,
        "text_results": text_results,
        "text_sse_fragment_gap_summary": text_gap_summaries,
        "image_results": image_results,
        "image_ttft_from_arrival_summary": summarize(image_ttft_values),
        "event_flush": flush_events(args.base_url),
        "limitations": [
            "Synthetic short replay; current-backbone content and cost policies are uncalibrated and keep all tokens.",
            "Client SSE content-fragment gaps and image TTFT are user-visible API metrics, not internal kernel timing.",
            "Not an answer-quality, pruning-benefit, formal SLO, or 480-run matrix result.",
        ],
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = {
        "text_statuses": [item.get("status") for item in text_results],
        "image_statuses": [item["request_result"].get("status") for item in image_results],
        "text_lane_count_at_image_arrival": [item["controller_decision"]["text_lane_count"] for item in image_results],
        "text_gap_summaries": text_gap_summaries,
        "image_ttft_from_arrival_summary": report["image_ttft_from_arrival_summary"],
        "output": str(output),
    }
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
