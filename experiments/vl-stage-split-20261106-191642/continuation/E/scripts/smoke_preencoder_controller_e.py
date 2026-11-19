#!/usr/bin/env python3
"""Exercise the uncalibrated live-controller output through one E image request."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from e_content_budget_policy import PreencoderBudgetConfig, raw_pixel_density_proxy
from e_live_joint_controller_e import TextDeadlineTracker, prepare_image_request_budget
from smoke_batch_multi_image_e import make_chart
from smoke_default_equivalence_e import flush_events, run_request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8285/v1/chat/completions")
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    asset_dir = Path(args.asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    image_bytes = make_chart("PRE-E-CONTROLLER", (512, 512), (35, 150, 95))
    image_sha = hashlib.sha256(image_bytes).hexdigest()
    image_path = asset_dir / "preencoder-controller-chart-512.png"
    image_path.write_bytes(image_bytes)
    image = Image.open(image_path).convert("RGB")

    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    processed = processor.image_processor(images=[image], return_tensors="pt")
    grid = [int(value) for value in processed["image_grid_thw"][0].tolist()]
    merge_size = int(getattr(processor.image_processor, "merge_size", 2))
    full_token_count = grid[0] * grid[1] * grid[2] // (merge_size * merge_size)
    rgb = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
    pixel_score = raw_pixel_density_proxy(rgb)

    tracker = TextDeadlineTracker()
    arrival_ns = time.monotonic_ns()
    decision = prepare_image_request_budget(
        query="Read the chart title and answer in one short sentence.",
        pixel_density_score=float(pixel_score),
        full_token_count=full_token_count,
        image_arrival_ns=arrival_ns,
        now_ns=arrival_ns,
        tracker=tracker,
        policy_config=PreencoderBudgetConfig(),  # intentionally uncalibrated
    )
    request = run_request(
        args.base_url,
        "preencoder-controller-uncalibrated-keepall",
        image_bytes,
        image_path.name,
        image_sha,
        "Read the chart title and answer in one short sentence.",
        decision["mm_processor_kwargs"],
    )
    event_flush = flush_events(args.base_url)
    record = {
        "experiment": "preencoder_controller_request_smoke",
        "formal_run": False,
        "calibrated_policy_expected": False,
        "image_sha256": image_sha,
        "processor_grid_thw": grid,
        "merge_size": merge_size,
        "full_decoder_tokens": full_token_count,
        "raw_pixel_density_score": pixel_score,
        "controller_decision": decision,
        "request": request,
        "event_flush": event_flush,
        "limitations": [
            "Calibration is absent, so the controller intentionally sent keep-all.",
            "One synthetic image request does not validate answer quality or live multi-lane control.",
        ],
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "http_status": request.get("status"),
                "keep_ratio": decision["request_keep_ratio"],
                "chosen_keep_count": decision["chosen_keep_count"],
                "full_token_count": full_token_count,
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
