#!/usr/bin/env python3
"""Send concurrent image requests with distinct per-request keep ratios."""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from smoke_batch_multi_image_e import data_url, make_chart
from smoke_batch_endpoint_image_pruning_e import flush_events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8285/v1/chat/completions")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--asset-dir", required=True)
    args = parser.parse_args()

    asset_dir = Path(args.asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        ("keep25", 0.25, "BUDGET-25", (30, 90, 190)),
        ("keep75", 0.75, "BUDGET-75", (180, 70, 40)),
    ]
    payloads = {}
    for label, ratio, title, color in cases:
        payload = make_chart(title, (768, 768), color)
        asset_path = asset_dir / f"{label}.png"
        asset_path.write_bytes(payload)
        payloads[label] = {
            "payload": payload,
            "asset": {
                "path": str(asset_path),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            },
        }

    barrier = threading.Barrier(len(cases))
    rows: list[dict] = []
    rows_lock = threading.Lock()

    def send_case(case: tuple[str, float, str, tuple[int, int, int]]) -> None:
        label, ratio, title, _color = case
        payload = payloads[label]["payload"]
        body = {
            "model": "qwen3vlpilot",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read the title of this chart."},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url(payload)},
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 8,
            "seed": 7710,
            "mm_processor_kwargs": {
                "images_kwargs": {"e_keep_ratio": ratio},
            },
        }
        request = urllib.request.Request(
            args.base_url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        barrier.wait(timeout=30)
        started_ns = time.monotonic_ns()
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.loads(response.read())
                row = {
                    "label": f"per-request-image-budget-{label}",
                    "keep_ratio": ratio,
                    "status": response.status,
                    "response_id": result.get("id"),
                    "usage": result.get("usage"),
                    "choices": [
                        {
                            "finish_reason": choice.get("finish_reason"),
                            "content": choice.get("message", {}).get("content"),
                        }
                        for choice in result.get("choices", [])
                    ],
                    "asset": payloads[label]["asset"],
                    "elapsed_client_ms": (time.monotonic_ns() - started_ns) / 1e6,
                }
        except urllib.error.HTTPError as error:
            row = {
                "label": f"per-request-image-budget-{label}",
                "keep_ratio": ratio,
                "status": error.code,
                "error_body": error.read().decode("utf-8", errors="replace")[:3000],
                "asset": payloads[label]["asset"],
            }
        with rows_lock:
            rows.append(row)

    threads = [threading.Thread(target=send_case, args=(case,)) for case in cases]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in sorted(rows, key=lambda item: item["keep_ratio"]):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.write(json.dumps(flush_events(args.base_url), ensure_ascii=False) + "\n")
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    print(f"wrote per-request budget results to {output}")


if __name__ == "__main__":
    main()
