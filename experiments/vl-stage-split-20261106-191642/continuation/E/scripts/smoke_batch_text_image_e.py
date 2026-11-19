#!/usr/bin/env python3
"""Exercise one batched text lane alongside two image conversations."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from smoke_batch_multi_image_e import data_url, make_chart
from smoke_batch_endpoint_image_pruning_e import flush_events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8285/v1/chat/completions/batch",
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--asset-dir", required=True)
    args = parser.parse_args()

    asset_dir = Path(args.asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    images = {
        "image-a": make_chart("MIX-A", (768, 768), (30, 90, 190)),
        "image-b": make_chart("MIX-B", (768, 768), (180, 70, 40)),
    }
    assets = {}
    for name, payload in images.items():
        path = asset_dir / f"{name}.png"
        path.write_bytes(payload)
        assets[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }

    conversations = [
        [
            {
                "role": "user",
                "content": "Explain the word steady in several complete sentences.",
            }
        ]
    ]
    for name, prompt in [
        ("image-a", "Read the title of this chart."),
        ("image-b", "Summarize the trend in this chart."),
    ]:
        conversations.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url(images[name])},
                        },
                    ],
                }
            ]
        )

    body = {
        "model": "qwen3vlpilot",
        "messages": conversations,
        "temperature": 0,
        "max_tokens": 32,
        "seed": 7703,
    }
    request = urllib.request.Request(
        args.base_url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_ns = time.monotonic_ns()
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read())
            batch_row = {
                "label": "one-text-two-images-batch-endpoint",
                "status": response.status,
                "batch_response_id": result.get("id"),
                "usage": result.get("usage"),
                "choices": [
                    {
                        "index": choice.get("index"),
                        "finish_reason": choice.get("finish_reason"),
                        "content": choice.get("message", {}).get("content"),
                    }
                    for choice in result.get("choices", [])
                ],
                "conversation_order": ["text-only", "image-a", "image-b"],
                "assets": assets,
                "elapsed_client_ms": (time.monotonic_ns() - started_ns) / 1e6,
            }
    except urllib.error.HTTPError as error:
        batch_row = {
            "label": "one-text-two-images-batch-endpoint",
            "status": error.code,
            "error_body": error.read().decode("utf-8", errors="replace")[:2000],
            "conversation_order": ["text-only", "image-a", "image-b"],
            "assets": assets,
        }

    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(batch_row, ensure_ascii=False) + "\n")
        stream.write(
            json.dumps(flush_events(args.base_url), ensure_ascii=False) + "\n"
        )
    print(json.dumps(batch_row, ensure_ascii=False))
    print(f"wrote batch result to {output}")


if __name__ == "__main__":
    main()
