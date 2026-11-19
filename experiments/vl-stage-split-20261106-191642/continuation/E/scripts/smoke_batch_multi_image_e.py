#!/usr/bin/env python3
"""Exercise same-batch, multi-image and aspect-ratio request joins."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw


def make_chart(label: str, size: tuple[int, int], color: tuple[int, int, int]) -> bytes:
    width, height = size
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, width - 9, height - 9), outline="black", width=3)
    draw.text((24, 24), f"E synthetic {label}", fill="black")
    bar_width = max(24, width // 12)
    baseline = height - max(48, height // 8)
    for idx, value in enumerate((0.25, 0.4, 0.58, 0.72, 0.9)):
        x = width // 8 + idx * (width // 7)
        top = int(baseline - value * (height * 0.48))
        draw.rectangle((x, top, x + bar_width, baseline), fill=color)
        draw.text((x, baseline + 8), f"{label}{idx + 1}", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def data_url(payload: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def request(
    base_url: str,
    label: str,
    text: str,
    images: list[tuple[str, bytes]],
    gate: threading.Barrier | None = None,
) -> dict:
    content: list[dict] = [{"type": "text", "text": text}]
    for _, payload in images:
        content.append({"type": "image_url", "image_url": {"url": data_url(payload)}})
    body = {
        "model": "qwen3vlpilot",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 8,
        "seed": 7701,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if gate is not None:
        gate.wait(timeout=10)
    started = time.monotonic_ns()
    response_id = None
    status = None
    usage = None
    finish_reason = None
    text_parts: list[str] = []
    try:
        req = urllib.request.Request(
            base_url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as response:
            status = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                chunk = json.loads(payload)
                response_id = chunk.get("id", response_id)
                if chunk.get("usage") is not None:
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        text_parts.append(delta["content"])
    except urllib.error.HTTPError as error:
        return {
            "label": label,
            "status": error.code,
            "elapsed_client_ms": (time.monotonic_ns() - started) / 1e6,
            "error_body": error.read().decode("utf-8", errors="replace")[:2000],
            "image_names": [name for name, _ in images],
        }
    return {
        "label": label,
        "status": status,
        "response_id": response_id,
        "usage": usage,
        "finish_reason": finish_reason,
        "elapsed_client_ms": (time.monotonic_ns() - started) / 1e6,
        "content": "".join(text_parts),
        "image_names": [name for name, _ in images],
    }


def flush_events(base_url: str) -> dict:
    root = base_url.removesuffix("/v1/chat/completions")
    req = urllib.request.Request(
        root + "/collective_rpc",
        data=json.dumps({"method": "pilot_flush_visual_events", "timeout": 30}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            return {"label": "cuda-event-flush", "status": response.status, "flush_result": json.loads(response.read())}
    except urllib.error.HTTPError as error:
        return {"label": "cuda-event-flush", "status": error.code, "error_body": error.read().decode(errors="replace")[:2000]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8285/v1/chat/completions")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--asset-dir", required=True)
    args = parser.parse_args()

    asset_dir = Path(args.asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    assets = {
        "square-a": make_chart("A", (768, 768), (30, 90, 190)),
        "square-b": make_chart("B", (768, 768), (180, 70, 40)),
        "landscape": make_chart("L", (768, 512), (40, 140, 80)),
    }
    asset_meta = {}
    for name, payload in assets.items():
        path = asset_dir / f"{name}.png"
        path.write_bytes(payload)
        asset_meta[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }

    gate = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(request, args.base_url, "concurrent-square-a", "Read the title of this chart.", [("square-a", assets["square-a"])], gate),
            pool.submit(request, args.base_url, "concurrent-square-b", "Summarize the trend in this chart.", [("square-b", assets["square-b"])], gate),
        ]
        records = [future.result() for future in futures]
    records.append(
        request(
            args.base_url,
            "one-request-two-images",
            "Compare the two charts and state one difference.",
            [("square-a", assets["square-a"]), ("square-b", assets["square-b"])],
        )
    )
    records.append(
        request(
            args.base_url,
            "landscape-aspect-ratio",
            "Read the title and summarize this landscape chart.",
            [("landscape", assets["landscape"])],
        )
    )
    records.append({"label": "synthetic-assets", "assets": asset_meta})
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
