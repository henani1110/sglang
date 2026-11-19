#!/usr/bin/env python3
"""Smoke the fixed-runtime streaming text-completion endpoint and trace flush."""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def post_json(url: str, body: dict, timeout: int = 180) -> tuple[int, bytes]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8285")
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    started_ns = time.monotonic_ns()
    response_id = None
    text_parts: list[str] = []
    finish_reason = None
    status = None
    error_body = None
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(
            {
                "model": "qwen3vlpilot",
                "prompt": "In one short sentence, explain why serving text and image requests together can cause interference.",
                "temperature": 0,
                "max_tokens": 16,
                "seed": 1702,
                "stream": True,
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
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
                for choice in chunk.get("choices", []):
                    text_parts.append(choice.get("text", ""))
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
    except urllib.error.HTTPError as error:
        status = error.code
        error_body = error.read().decode("utf-8", errors="replace")[:3000]

    elapsed_ms = (time.monotonic_ns() - started_ns) / 1e6
    flush_status, flush_bytes = post_json(
        args.base_url.rstrip("/") + "/collective_rpc",
        {"method": "pilot_flush_visual_events", "timeout": 30},
        timeout=45,
    )
    try:
        flush_result = json.loads(flush_bytes)
    except json.JSONDecodeError:
        flush_result = {"body": flush_bytes.decode("utf-8", errors="replace")[:3000]}

    record = {
        "label": "text-completion-stream",
        "status": status,
        "response_id": response_id,
        "elapsed_client_ms": elapsed_ms,
        "finish_reason": finish_reason,
        "text": "".join(text_parts),
        "error_body": error_body,
    }
    flush_record = {
        "label": "cuda-event-flush-after-text-completion",
        "status": flush_status,
        "flush_result": flush_result,
    }
    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.write(json.dumps(flush_record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False))
    print(json.dumps(flush_record, ensure_ascii=False))


if __name__ == "__main__":
    main()
