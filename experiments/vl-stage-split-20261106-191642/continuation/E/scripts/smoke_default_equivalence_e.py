#!/usr/bin/env python3
"""Compare native/explicit processor defaults and exercise 512/768 pixel budgets."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoProcessor

from smoke_batch_multi_image_e import make_chart


def run_request(
    base_url: str,
    label: str,
    image_data: bytes,
    image_name: str,
    image_sha256: str,
    prompt: str,
    mm_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "qwen3vlpilot",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(image_data).decode("ascii")
                        },
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 96,
        "seed": 17701,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if mm_kwargs is not None:
        body["mm_processor_kwargs"] = mm_kwargs
    req = urllib.request.Request(
        base_url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    content: list[str] = []
    usage = None
    response_id = None
    finish_reason = None
    try:
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
                        content.append(delta["content"])
    except urllib.error.HTTPError as error:
        return {
            "label": label,
            "status": error.code,
            "error_body": error.read().decode("utf-8", errors="replace")[:2000],
            "image_name": image_name,
            "image_sha256": image_sha256,
            "mm_processor_kwargs": mm_kwargs,
        }
    return {
        "label": label,
        "status": status,
        "response_id": response_id,
        "usage": usage,
        "finish_reason": finish_reason,
        "content": "".join(content),
        "image_name": image_name,
        "image_sha256": image_sha256,
        "mm_processor_kwargs": mm_kwargs,
    }


def flush_events(base_url: str) -> dict[str, Any]:
    root = base_url.removesuffix("/v1/chat/completions")
    req = urllib.request.Request(
        root + "/collective_rpc",
        data=json.dumps({"method": "pilot_flush_visual_events", "timeout": 30}).encode(),
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
            "error_body": error.read().decode("utf-8", errors="replace")[:2000],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8285/v1/chat/completions")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--asset-dir", required=True)
    args = parser.parse_args()

    asset_dir = Path(args.asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    image_assets = {
        "square-768": make_chart("D768", (768, 768), (35, 100, 185)),
        "square-512": make_chart("D512", (512, 512), (35, 150, 95)),
    }
    asset_meta: dict[str, dict[str, Any]] = {}
    for name, payload in image_assets.items():
        path = asset_dir / f"{name}.png"
        path.write_bytes(payload)
        asset_meta[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }

    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
    )
    processor_size = getattr(processor.image_processor, "size", {})
    explicit_default_size = {
        str(key): int(value)
        for key, value in dict(processor_size).items()
        if value is not None
    }
    required_keys = {"shortest_edge", "longest_edge"}
    if not required_keys.issubset(explicit_default_size):
        raise SystemExit(f"processor size does not include both explicit edges: {explicit_default_size}")

    processor_rows = []
    for name, payload in image_assets.items():
        output = processor.image_processor(
            images=[Image.open(io.BytesIO(payload)).convert("RGB")],
            return_tensors="pt",
        )
        grid = [int(x) for x in output["image_grid_thw"][0].tolist()]
        merge_size = int(getattr(processor.image_processor, "merge_size", 2))
        processor_rows.append(
            {
                "image_name": name,
                "image_grid_thw": grid,
                "merged_visual_tokens": grid[0] * grid[1] * grid[2] // (merge_size * merge_size),
                "pixel_values_shape": [int(x) for x in output["pixel_values"].shape],
            }
        )

    explicit_default = {"images_kwargs": {"size": explicit_default_size}}
    prompt = "Read the chart title and answer in one short sentence."
    records: list[dict[str, Any]] = [
        {
            "label": "processor-reference",
            "processor_class": type(processor).__name__,
            "image_processor_class": type(processor.image_processor).__name__,
            "explicit_default_size": explicit_default_size,
            "image_grids": processor_rows,
            "assets": asset_meta,
        }
    ]
    image_768 = image_assets["square-768"]
    hash_768 = asset_meta["square-768"]["sha256"]
    image_512 = image_assets["square-512"]
    hash_512 = asset_meta["square-512"]["sha256"]
    records.append(run_request(args.base_url, "native-default-768", image_768, "square-768", hash_768, prompt, None))
    records.append(run_request(args.base_url, "explicit-default-768", image_768, "square-768", hash_768, prompt, explicit_default))
    records.append(run_request(args.base_url, "native-default-512", image_512, "square-512", hash_512, prompt, None))
    records.append(run_request(args.base_url, "explicit-default-512", image_512, "square-512", hash_512, prompt, explicit_default))
    records.append(
        run_request(
            args.base_url,
            "cap-262144-768",
            image_768,
            "square-768",
            hash_768,
            prompt,
            {"images_kwargs": {"size": {"shortest_edge": 3136, "longest_edge": 262144}}},
        )
    )
    records.append(
        run_request(
            args.base_url,
            "cap-65536-768",
            image_768,
            "square-768",
            hash_768,
            prompt,
            {"images_kwargs": {"size": {"shortest_edge": 3136, "longest_edge": 65536}}},
        )
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
