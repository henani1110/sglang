#!/usr/bin/env python3
"""One-image E diagnostic for graceful shutdown and request/event cleanup."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from smoke_batch_multi_image_e import make_chart
from smoke_default_equivalence_e import flush_events, run_request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8285/v1/chat/completions")
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    asset_dir = Path(args.asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    payload = make_chart("SHUTDOWN-15S", (512, 512), (35, 150, 95))
    image_hash = hashlib.sha256(payload).hexdigest()
    image_path = asset_dir / "shutdown-timeout-chart-512.png"
    image_path.write_bytes(payload)
    result = run_request(
        args.base_url,
        "single-image-keepall",
        payload,
        image_path.name,
        image_hash,
        "Read the chart title and answer in one short sentence.",
        {"images_kwargs": {"e_keep_ratio": 1.0}},
    )
    flush = flush_events(args.base_url)
    record = {
        "experiment": "shutdown_timeout_image_diagnostic",
        "formal_run": False,
        "server_shutdown_timeout_seconds": 15,
        "image_path": str(image_path),
        "image_sha256": image_hash,
        "input_size": [512, 512],
        "request": result,
        "flush": flush,
        "limitations": [
            "Synthetic image and keep-all request only; not answer-quality, pruning, latency or goodput evidence.",
            "Shutdown behavior must be compared with warning-bearing default-timeout runs before inferring causality.",
        ],
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result.get("status"), "flush": flush, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
