#!/usr/bin/env python3
"""Send two synthetic image conversations through vLLM's batch endpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from smoke_batch_multi_image_e import data_url, make_chart


def flush_events(base_url: str) -> dict:
    root = base_url.split("/v1/chat/completions", 1)[0]
    request = urllib.request.Request(
        root + "/collective_rpc",
        data=json.dumps(
            {"method": "pilot_flush_visual_events", "timeout": 30}
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return {
                "label": "cuda-event-flush",
                "status": response.status,
                "flush_result": json.loads(response.read()),
            }
    except urllib.error.HTTPError as error:
        return {
            "label": "cuda-event-flush",
            "status": error.code,
            "error_body": error.read().decode("utf-8", errors="replace")[:2000],
        }


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
        "batch-a": make_chart("BATCH-A", (768, 768), (30, 90, 190)),
        "batch-b": make_chart("BATCH-B", (768, 768), (180, 70, 40)),
    }
    metadata = {}
    for name, payload in images.items():
        path = asset_dir / f"{name}.png"
        path.write_bytes(payload)
        metadata[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }

    texts = ["Read the title of this chart.", "Summarize the trend in this chart."]
    conversations = []
    for image_name, text in zip(images, texts):
        content = [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": data_url(images[image_name])},
            },
        ]
        conversations.append([{"role": "user", "content": content}])

    body = {
        "model": "qwen3vlpilot",
        "messages": conversations,
        "temperature": 0,
        "max_tokens": 8,
        "seed": 7702,
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
                "label": "two-image-batch-endpoint",
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
                "input_image_order": list(images),
                "assets": metadata,
                "elapsed_client_ms": (time.monotonic_ns() - started_ns) / 1e6,
            }
    except urllib.error.HTTPError as error:
        batch_row = {
            "label": "two-image-batch-endpoint",
            "status": error.code,
            "error_body": error.read().decode("utf-8", errors="replace")[:2000],
            "input_image_order": list(images),
            "assets": metadata,
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
