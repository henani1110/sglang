#!/usr/bin/env python3
"""Exercise the E-only D-style S split with synthetic image plus active text streams."""
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
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw


def make_chart() -> bytes:
    image = Image.new("RGB", (768, 768), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 728, 728), outline="black", width=4)
    draw.text((75, 70), "Synthetic E stage split", fill="black")
    for idx, height in enumerate((140, 220, 310, 405, 520)):
        x = 130 + idx * 105
        draw.rectangle((x, 650 - height, x + 54, 650), fill=(30, 80 + idx * 24, 180))
        draw.text((x, 665), f"Q{idx + 1}", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def stream_request(
    url: str,
    label: str,
    body: dict,
    first_content: threading.Event | None = None,
    test_start_ns: int | None = None,
    lane_start_callback: Callable[[str, int], None] | None = None,
    token_callback: Callable[[str, int], None] | None = None,
    lane_finish_callback: Callable[[str], None] | None = None,
    request_arrival_ns: int | None = None,
) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_ns = time.monotonic_ns()
    client_start_ns = started_ns if request_arrival_ns is None else int(request_arrival_ns)
    if lane_start_callback is not None:
        lane_start_callback(label, started_ns)
    status: int | None = None
    response_id: str | None = None
    finish_reason: str | None = None
    usage = None
    parts: list[str] = []
    token_times_ms: list[float] = []
    error_body = None
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
                    # Chat completions use delta.content; the completions API
                    # used for the text lanes streams choice.text instead.
                    token = delta.get("content") or choice.get("text")
                    if token:
                        parts.append(token)
                        now_ns = time.monotonic_ns()
                        token_times_ms.append((now_ns - client_start_ns) / 1e6)
                        if token_callback is not None:
                            token_callback(label, now_ns)
                        if first_content is not None:
                            first_content.set()
    except urllib.error.HTTPError as error:
        status = error.code
        error_body = error.read().decode("utf-8", errors="replace")[:3000]
    except Exception as error:  # retain client-side failure evidence in the output
        error_body = f"{type(error).__name__}: {error}"

    ended_ns = time.monotonic_ns()
    if lane_finish_callback is not None:
        lane_finish_callback(label)
    return {
        "kind": label,
        "status": status,
        "response_id": response_id,
        "test_start_offset_ms": (
            (started_ns - test_start_ns) / 1e6 if test_start_ns is not None else None
        ),
        "queue_wait_ms": (started_ns - client_start_ns) / 1e6,
        "elapsed_client_ms": (ended_ns - client_start_ns) / 1e6,
        "finish_reason": finish_reason,
        "usage": usage,
        "content": "".join(parts),
        "client_token_times_ms": token_times_ms,
        "client_token_gaps_ms": [
            round(right - left, 3)
            for left, right in zip(token_times_ms, token_times_ms[1:])
        ],
        "error_body": error_body,
    }


def flush_events(root_url: str) -> dict:
    req = urllib.request.Request(
        root_url.rstrip("/") + "/collective_rpc",
        data=json.dumps({"method": "pilot_flush_visual_events", "timeout": 30}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            return {"kind": "flush", "status": response.status, "body": json.loads(response.read())}
    except Exception as error:
        return {"kind": "flush", "status": None, "error": f"{type(error).__name__}: {error}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8285")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--asset-path", required=True)
    parser.add_argument("--text-lanes", type=int, default=2)
    args = parser.parse_args()
    if args.text_lanes < 1 or args.text_lanes > 6:
        raise SystemExit("--text-lanes must be in [1, 6]")

    image_bytes = make_chart()
    image_path = Path(args.asset_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(image_bytes)
    image_data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    image_body = {
        "model": "qwen3vlpilot",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Read the trend and chart title in one sentence."},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]}],
        "temperature": 0,
        "max_tokens": 32,
        "seed": 1709,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    test_start_ns = time.monotonic_ns()
    first_content = [threading.Event() for _ in range(args.text_lanes)]
    executor = ThreadPoolExecutor(max_workers=args.text_lanes + 1)
    futures = []
    image_sent_after_text_first_content = False
    gate_failure = None
    try:
        for lane in range(args.text_lanes):
            text_body = {
                "model": "qwen3vlpilot",
                "prompt": (
                    "Continue a numbered technical design checklist. Give one short, "
                    "distinct item per line and keep going until the token limit. "
                    "1. Keep scheduler decisions observable."
                ),
                "temperature": 0,
                "max_tokens": 96,
                "seed": 1710 + lane,
                "stream": True,
            }
            futures.append(executor.submit(
                stream_request,
                args.base_url.rstrip("/") + "/v1/completions",
                f"text-{lane + 1}",
                text_body,
                first_content[lane],
                test_start_ns,
            ))

        deadline = time.monotonic() + 45
        for lane_index, event in enumerate(first_content):
            if not event.wait(max(0.0, deadline - time.monotonic())):
                missing = [
                    index + 1 for index, lane_event in enumerate(first_content)
                    if not lane_event.is_set()
                ]
                gate_failure = {
                    "kind": "gate_failure",
                    "error": "not all text lanes produced first content before image arrival",
                    "timed_out_while_waiting_for_lane": lane_index + 1,
                    "missing_lanes": missing,
                    "timeout_seconds": 45,
                }
                break

        if gate_failure is None:
            image_future = executor.submit(
                stream_request,
                args.base_url.rstrip("/") + "/v1/chat/completions",
                "image",
                image_body,
                None,
                test_start_ns,
            )
            futures.append(image_future)
            image_sent_after_text_first_content = True
    finally:
        executor.shutdown(wait=True, cancel_futures=False)

    records = []
    for future in futures:
        try:
            records.append(future.result())
        except Exception as error:  # retain unexpected worker failures too
            records.append({
                "kind": "client_worker_failure",
                "error": f"{type(error).__name__}: {error}",
            })
    if gate_failure is not None:
        records.append(gate_failure)

    records.append({
        "kind": "synthetic_asset",
        "path": str(image_path),
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
        "bytes": len(image_bytes),
        "text_lanes": args.text_lanes,
        "image_sent_after_text_first_content": image_sent_after_text_first_content,
    })
    records.append(flush_events(args.base_url))
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    for record in records:
        print(json.dumps(record, ensure_ascii=False))
    print(f"wrote {len(records)} rows to {output_path}")
    if gate_failure is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
