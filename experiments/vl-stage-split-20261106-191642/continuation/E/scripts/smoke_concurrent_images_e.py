#!/usr/bin/env python3
"""Launch a simultaneous image-request burst to capture encoder co-batching."""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from smoke_batch_multi_image_e import flush_events, make_chart, request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8285/v1/chat/completions")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if args.concurrency < 2 or args.concurrency > 8:
        raise SystemExit("--concurrency must be between 2 and 8")

    asset_dir = Path(args.asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    palette = [
        (30, 90, 190),
        (180, 70, 40),
        (40, 140, 80),
        (130, 70, 170),
        (190, 140, 30),
        (30, 150, 160),
        (180, 60, 110),
        (80, 100, 130),
    ]
    assets = []
    for idx in range(args.concurrency):
        name = f"burst-{idx + 1}"
        payload = make_chart(f"R{idx + 1}", (768, 768), palette[idx])
        path = asset_dir / f"{name}.png"
        path.write_bytes(payload)
        assets.append((name, payload))

    gate = threading.Barrier(args.concurrency)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                request,
                args.base_url,
                f"simultaneous-{name}",
                "Summarize the rising trend in this chart.",
                [(name, payload)],
                gate,
            )
            for name, payload in assets
        ]
        records = [future.result() for future in futures]
    records.append(
        {
            "label": "synthetic-assets",
            "assets": [
                {
                    "name": name,
                    "path": str(asset_dir / f"{name}.png"),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
                for name, payload in assets
            ],
        }
    )
    records.append(flush_events(args.base_url))
    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    for record in records:
        print(json.dumps(record, ensure_ascii=False))
    print(f"wrote {len(records)} records to {output}")


if __name__ == "__main__":
    main()
