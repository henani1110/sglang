#!/usr/bin/env python3
"""Run a fixed synthetic image stream and record first-content/full-response times."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def post_request(
    base_url: str,
    image_bytes: bytes,
    image_sha256: str,
    request_index: int,
    warmup: bool,
    seed: int,
    max_tokens: int,
) -> dict[str, Any]:
    body = {
        "model": "qwen3vlpilot",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Read the chart title. Answer with only the exact title."},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
                }},
            ],
        }],
        "temperature": 0,
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_ns = time.perf_counter_ns()
    headers_ns: int | None = None
    first_sse_ns: int | None = None
    first_content_ns: int | None = None
    content_parts: list[str] = []
    response_id = None
    finish_reason = None
    usage = None
    status = None
    error_body = None
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            headers_ns = time.perf_counter_ns()
            status = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                if first_sse_ns is None:
                    first_sse_ns = time.perf_counter_ns()
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
                    piece = delta.get("content")
                    if piece:
                        if first_content_ns is None:
                            first_content_ns = time.perf_counter_ns()
                        content_parts.append(piece)
    except urllib.error.HTTPError as exc:
        status = exc.code
        error_body = exc.read().decode("utf-8", errors="replace")[:2000]
    except Exception as exc:  # Keep failed attempts in the calibration record.
        error_body = f"{type(exc).__name__}: {exc}"
    finished_ns = time.perf_counter_ns()

    def elapsed_ms(stamp: int | None) -> float | None:
        return None if stamp is None else (stamp - started_ns) / 1_000_000

    return {
        "kind": "request_timing",
        "request_index": request_index,
        "warmup": warmup,
        "status": status,
        "response_id": response_id,
        "http_headers_ms": elapsed_ms(headers_ns),
        "first_sse_ms": elapsed_ms(first_sse_ns),
        "first_content_ms": elapsed_ms(first_content_ns),
        "full_response_ms": elapsed_ms(finished_ns),
        "finish_reason": finish_reason,
        "usage": usage,
        "content": "".join(content_parts),
        "image_sha256": image_sha256,
        "error": error_body,
        "request_settings": {
            "seed": seed,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": True,
        },
    }


def flush_server(base_url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/collective_rpc",
        data=json.dumps({"method": "pilot_flush_visual_events", "timeout": 30}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return {"kind": "trace_flush", "status": response.status, "body": json.loads(response.read())}
    except Exception as exc:
        return {"kind": "trace_flush", "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("probe-off", "queue-only", "full-probe"), required=True)
    parser.add_argument("--block-id", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8285")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--warmup-count", type=int, default=2)
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17701)
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()
    if args.warmup_count < 0 or args.sample_count <= 0:
        raise SystemExit("warmup-count must be non-negative and sample-count must be positive")
    image_bytes = args.image.read_bytes()
    image_sha = hashlib.sha256(image_bytes).hexdigest()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records = [{
        "kind": "calibration_metadata",
        "profile": args.profile,
        "block_id": args.block_id,
        "image_path": str(args.image),
        "image_sha256": image_sha,
        "warmup_count": args.warmup_count,
        "sample_count": args.sample_count,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "start_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }]
    with args.output_jsonl.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(records[0], ensure_ascii=False) + "\n")
    for index in range(args.warmup_count + args.sample_count):
        row = post_request(
            args.base_url,
            image_bytes,
            image_sha,
            index,
            index < args.warmup_count,
            args.seed,
            args.max_tokens,
        )
        row["profile"] = args.profile
        row["block_id"] = args.block_id
        records.append(row)
        with args.output_jsonl.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({
            "profile": args.profile,
            "request_index": index,
            "warmup": index < args.warmup_count,
            "status": row["status"],
            "first_content_ms": row["first_content_ms"],
            "full_response_ms": row["full_response_ms"],
        }, ensure_ascii=False), flush=True)
    records.append(flush_server(args.base_url))
    records[-1]["profile"] = args.profile
    records[-1]["block_id"] = args.block_id
    with args.output_jsonl.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(records[-1], ensure_ascii=False) + "\n")
    print(json.dumps(records[-1], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
