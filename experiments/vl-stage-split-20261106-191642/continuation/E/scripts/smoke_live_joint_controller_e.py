#!/usr/bin/env python3
"""Exercise live text-slack tracking before admitting one E image request."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from e_content_budget_policy import PreencoderBudgetConfig, raw_pixel_density_proxy
from e_live_joint_controller_e import TextDeadlineTracker, prepare_image_request_budget
from smoke_batch_multi_image_e import make_chart
from smoke_stage_split_e import flush_events, stream_request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8285")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--asset-path", required=True)
    parser.add_argument("--text-lanes", type=int, default=2)
    parser.add_argument("--first-token-timeout-seconds", type=float, default=45.0)
    args = parser.parse_args()
    if not 1 <= args.text_lanes <= 6:
        raise SystemExit("--text-lanes must be in [1, 6]")

    asset_path = Path(args.asset_path)
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    image_bytes = make_chart("J", (768, 768), (40, 140, 80))
    image_sha = hashlib.sha256(image_bytes).hexdigest()
    asset_path.write_bytes(image_bytes)
    image = Image.open(asset_path).convert("RGB")
    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    processed = processor.image_processor(images=[image], return_tensors="pt")
    grid = [int(value) for value in processed["image_grid_thw"][0].tolist()]
    merge_size = int(getattr(processor.image_processor, "merge_size", 2))
    full_tokens = grid[0] * grid[1] * grid[2] // (merge_size * merge_size)
    rgb = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
    pixel_score = float(raw_pixel_density_proxy(rgb))
    query = "Read the trend and chart title in one sentence."

    tracker = TextDeadlineTracker()
    first_content = [Event() for _ in range(args.text_lanes)]
    test_start_ns = time.monotonic_ns()
    executor = ThreadPoolExecutor(max_workers=args.text_lanes + 1)
    futures = []
    gate_failure = None
    decision = None
    image_result = None
    try:
        for lane in range(args.text_lanes):
            lane_label = f"text-{lane + 1}"
            text_body = {
                "model": "qwen3vlpilot",
                "prompt": (
                    "Continue a numbered technical design checklist. Give one short, "
                    "distinct item per line and keep going until the token limit. "
                    "1. Keep scheduler decisions observable."
                ),
                "temperature": 0,
                "max_tokens": 96,
                "seed": 1810 + lane,
                "stream": True,
            }
            futures.append(
                executor.submit(
                    stream_request,
                    args.base_url.rstrip("/") + "/v1/completions",
                    lane_label,
                    text_body,
                    first_content[lane],
                    test_start_ns,
                    lambda label, started: tracker.record_token(label, started),
                    lambda label, token_ns: tracker.record_token(label, token_ns),
                    lambda label: tracker.finish_lane(label),
                )
            )

        deadline = time.monotonic() + args.first_token_timeout_seconds
        for index, ready in enumerate(first_content):
            if not ready.wait(max(0.0, deadline - time.monotonic())):
                gate_failure = {
                    "error": "not all text lanes produced first content before image arrival",
                    "waiting_lane": index + 1,
                    "missing_lanes": [i + 1 for i, event in enumerate(first_content) if not event.is_set()],
                }
                break

        if gate_failure is None:
            image_arrival_ns = time.monotonic_ns()
            decision_now_ns = time.monotonic_ns()
            decision = prepare_image_request_budget(
                query=query,
                pixel_density_score=pixel_score,
                full_token_count=full_tokens,
                image_arrival_ns=image_arrival_ns,
                now_ns=decision_now_ns,
                tracker=tracker,
                text_max_gap_ms=30.0,
                image_ttft_ms=55.0,
                policy_config=PreencoderBudgetConfig(),  # intentionally uncalibrated
            )
            image_data_url = "data:image/png;base64," + __import__("base64").b64encode(image_bytes).decode("ascii")
            image_body = {
                "model": "qwen3vlpilot",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ]}],
                "temperature": 0,
                "max_tokens": 32,
                "seed": 1709,
                "stream": True,
                "stream_options": {"include_usage": True},
                "mm_processor_kwargs": decision["mm_processor_kwargs"],
            }
            image_result = stream_request(
                args.base_url.rstrip("/") + "/v1/chat/completions",
                "image",
                image_body,
                None,
                test_start_ns,
            )
    finally:
        executor.shutdown(wait=True, cancel_futures=False)

    text_results = []
    for future in futures:
        try:
            text_results.append(future.result())
        except Exception as error:
            text_results.append({"kind": "client_worker_failure", "error": f"{type(error).__name__}: {error}"})
    flush = flush_events(args.base_url) if gate_failure is None else None
    record = {
        "experiment": "live_joint_controller_text_image_smoke",
        "formal_run": False,
        "text_lanes_requested": args.text_lanes,
        "image_sent_after_all_text_first_content": gate_failure is None,
        "gate_failure": gate_failure,
        "image_sha256": image_sha,
        "processor_grid_thw": grid,
        "merge_size": merge_size,
        "full_decoder_tokens": full_tokens,
        "pixel_density_score": pixel_score,
        "controller_decision": decision,
        "text_results": text_results,
        "image_result": image_result,
        "event_flush": flush,
        "limitations": [
            "Uncalibrated policy must keep all visual tokens; this smoke checks live timestamp plumbing and request metadata only.",
            "Synthetic short run; not a quality, latency, or goodput result and not part of the 480 formal runs.",
        ],
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "image_sent": gate_failure is None,
                "active_text_lanes_at_decision": decision["text_lane_count"] if decision else None,
                "text_slack_ms": decision["text_slack_raw_ms"] if decision else None,
                "image_keep_ratio": decision["request_keep_ratio"] if decision else None,
                "text_http": [item.get("status") for item in text_results],
                "image_http": image_result.get("status") if image_result else None,
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )
    if gate_failure is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
