#!/usr/bin/env python3
"""End-to-end synthetic-image smoke for per-request Qwen3-VL resolution budgets."""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw


def make_image() -> bytes:
    image = Image.new("RGB", (768, 768), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 728, 728), outline="black", width=5)
    for index, value in enumerate((120, 245, 375, 510, 640)):
        x = 120 + index * 115
        draw.rectangle((x, 420 - value // 2, x + 48, 420), fill=(35, 90 + index * 20, 175))
        draw.text((x - 8, 440), f"C{index + 1}", fill="black")
    draw.text((80, 70), "Synthetic E processor budget smoke", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def request(url: str, image_data: str, label: str, mm_kwargs: dict | None) -> dict:
    body = {
        "model": "qwen3vlpilot",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this synthetic chart in one short sentence."},
                    {"type": "image_url", "image_url": {"url": image_data}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 32,
        "seed": 1701,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if mm_kwargs is not None:
        body["mm_processor_kwargs"] = mm_kwargs
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic_ns()
    content_parts: list[str] = []
    token_times_ms: list[float] = []
    usage = None
    finish_reason = None
    response_id = None
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            status = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]" or not payload:
                    continue
                chunk = json.loads(payload)
                response_id = chunk.get("id", response_id)
                if chunk.get("usage") is not None:
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
                    delta = choice.get("delta", {})
                    text_piece = delta.get("content")
                    if text_piece:
                        content_parts.append(text_piece)
                        token_times_ms.append((time.monotonic_ns() - started) / 1e6)
    except urllib.error.HTTPError as error:
        return {
            "label": label,
            "status": error.code,
            "elapsed_client_ms": (time.monotonic_ns() - started) / 1e6,
            "error_body": error.read().decode(errors="replace")[:2000],
            "mm_processor_kwargs": mm_kwargs,
        }
    elapsed_ms = (time.monotonic_ns() - started) / 1e6
    return {
        "label": label,
        "status": status,
        "elapsed_client_ms": elapsed_ms,
        "usage": usage,
        "finish_reason": finish_reason,
        "content": "".join(content_parts),
        "response_id": response_id,
        "client_token_times_ms": token_times_ms,
        "client_token_gaps_ms": [
            round(b - a, 3) for a, b in zip(token_times_ms, token_times_ms[1:])
        ],
        "mm_processor_kwargs": mm_kwargs,
    }


def flush_visual_events(base_url: str) -> dict:
    root = base_url.removesuffix("/v1/chat/completions")
    req = urllib.request.Request(
        root + "/collective_rpc",
        data=json.dumps(
            {"method": "pilot_flush_visual_events", "timeout": 30}
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            return {
                "label": "cuda-event-flush",
                "status": response.status,
                "flush_result": json.loads(response.read()),
            }
    except urllib.error.HTTPError as error:
        return {
            "label": "cuda-event-flush",
            "status": error.code,
            "error_body": error.read().decode(errors="replace")[:2000],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8285/v1/chat/completions")
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    png = make_image()
    data_url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    cases = [
        ("native-default", None),
        (
            "cap-262144",
            {"images_kwargs": {"size": {"shortest_edge": 3136, "longest_edge": 262144}}},
        ),
        (
            "cap-65536",
            {"images_kwargs": {"size": {"shortest_edge": 3136, "longest_edge": 65536}}},
        ),
    ]
    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = [request(args.base_url, data_url, label, kwargs) for label, kwargs in cases]
    records.append(flush_visual_events(args.base_url))
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    for record in records:
        print(json.dumps(record, ensure_ascii=False))
    print(f"wrote {len(records)} records to {output}")


if __name__ == "__main__":
    main()
