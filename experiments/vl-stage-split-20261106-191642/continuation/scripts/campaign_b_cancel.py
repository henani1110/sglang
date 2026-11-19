#!/usr/bin/env python3
"""Abort one image stream after its first output token under S and retain evidence."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

RUN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RUN_ROOT / "scripts"))
from trace_client import IMAGE_OUTPUT_TOKENS, QUESTION, Pilot, append_jsonl, decode_sse_ids


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--control-file", required=True)
    parser.add_argument("--client-events", required=True)
    parser.add_argument("--diagnostics-file", required=True)
    parser.add_argument("--equivalence-file", required=True)
    parser.add_argument("--results-file", required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--episodes", type=int, default=1)  # accepted by the shared cell runner
    parser.add_argument("--text-tokens", type=int, default=192)
    args = parser.parse_args()

    pilot = Pilot(SimpleNamespace(
        base_url=args.base_url,
        client_events=args.client_events,
        control_file=args.control_file,
        image_dir=args.image_dir,
        model_path=args.model_path,
        model_name="qwen3vlpilot",
        text_tokens=args.text_tokens,
        arrival_offset_ms=5.0,
        diagnostics_file=args.diagnostics_file,
        equivalence_file=args.equivalence_file,
        results_file=args.results_file,
    ))
    episode_id = f"{args.cell}-block1-R-cancel"
    try:
        health = await pilot.client.get(args.base_url.rstrip("/") + "/health")
        health.raise_for_status()
        pilot.set_mode("R", arm="R", episode_id=episode_id, image_index=0)
        await pilot.reset_caches()

        token_counts = [0, 0, 0, 0]
        ready_events = [asyncio.Event() for _ in range(4)]
        trigger = asyncio.Event()
        trigger_state: dict[str, object] = {}
        text_results: list[dict[str, object] | None] = [None, None, None, None]

        async def run_text(lane: int) -> None:
            label = f"{episode_id}-text{lane}"
            payload = {
                "model": pilot.args.model_name,
                "prompt": pilot.prompts[lane],
                "max_tokens": args.text_tokens,
                "temperature": 0,
                "ignore_eos": True,
                "stream": True,
                "return_token_ids": True,
                "user": label,
            }

            def on_token(count: int, time_ns: int) -> bool:
                token_counts[lane] += count
                if (
                    not trigger.is_set()
                    and token_counts[0] >= 16
                    and all(value > 0 for value in token_counts[1:])
                ):
                    trigger_state["time_ns"] = time_ns
                    trigger_state["counts"] = list(token_counts)
                    trigger.set()
                    return True
                return False

            text_results[lane] = await pilot.read_stream(
                "/v1/completions",
                payload,
                request_label=label,
                lane=lane,
                ready=ready_events[lane],
                on_token=on_token,
            )

        text_tasks = [asyncio.create_task(run_text(lane)) for lane in range(4)]
        await asyncio.wait_for(trigger.wait(), timeout=30)
        trigger_ns = int(trigger_state["time_ns"])
        target_ns = trigger_ns + 5_000_000
        if target_ns > time.monotonic_ns():
            await asyncio.sleep((target_ns - time.monotonic_ns()) / 1e9)
        anchor_ns = time.monotonic_ns()
        client_rows = [{
            "kind": "episode_anchor",
            "episode_id": episode_id,
            "block": f"{args.cell}-block1",
            "arm": "R",
            "time_ns": anchor_ns,
            "trigger_ns": trigger_ns,
            "arrival_offset_ms": (anchor_ns - trigger_ns) / 1e6,
            "text_token_counts_at_trigger": trigger_state["counts"],
            "text_token_counts_at_anchor": list(token_counts),
            "active_text_streams_at_anchor": [not task.done() for task in text_tasks],
            "image_index": 0,
            "cancel_test": True,
        }]

        image_label = f"{episode_id}-image"
        image_body = {
            "model": pilot.args.model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": pilot.image_data_urls[0]}},
                    {"type": "text", "text": QUESTION},
                ],
            }],
            "max_tokens": IMAGE_OUTPUT_TOKENS,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "return_token_ids": True,
            "user": image_label,
        }
        image_first_token = asyncio.Event()
        image_state: dict[str, object] = {"request_id": None, "token_ids": [], "started_ns": None}
        image_rows: list[dict[str, object]] = []

        async def stream_image_until_cancelled() -> None:
            response_id = None
            chunk_index = 0
            try:
                async with pilot.client.stream(
                    "POST", pilot.base + "/v1/chat/completions", json=image_body
                ) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode(errors="replace")
                        raise RuntimeError(f"HTTP {response.status_code}: {body[:1000]}")
                    image_state["started_ns"] = time.monotonic_ns()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        raw = line[6:]
                        if raw == "[DONE]":
                            image_rows.append({"kind": "sse_done", "time_ns": time.monotonic_ns()})
                            break
                        now = time.monotonic_ns()
                        obj = json.loads(raw)
                        response_id = response_id or obj.get("id")
                        ids, prompt_ids, text = decode_sse_ids(obj, "/v1/chat/completions")
                        image_state["request_id"] = response_id
                        if ids:
                            image_state["token_ids"].extend(ids)
                        image_rows.append({
                            "kind": "sse_chunk",
                            "time_ns": now,
                            "chunk_index": chunk_index,
                            "request_id": response_id,
                            "lane": None,
                            "token_ids": ids,
                            "prompt_token_ids": prompt_ids,
                            "text": text,
                            "finish_reason": (obj.get("choices") or [{}])[0].get("finish_reason"),
                            "raw": raw,
                        })
                        chunk_index += 1
                        if ids and not image_first_token.is_set():
                            image_first_token.set()
                            await asyncio.Future()
            finally:
                now = time.monotonic_ns()
                for row in image_rows:
                    row.update(episode_id=episode_id, arm="R", block=f"{args.cell}-block1", trial=0)
                client_rows.extend(image_rows)
                client_rows.append({
                    "kind": "client_stream_cancel",
                    "episode_id": episode_id,
                    "arm": "R",
                    "block": f"{args.cell}-block1",
                    "time_ns": now,
                    "request_id": image_state.get("request_id"),
                    "partial_image_tokens": len(image_state.get("token_ids", [])),
                })
                append_jsonl(Path(args.client_events), client_rows)

        image_task = asyncio.create_task(stream_image_until_cancelled())
        await asyncio.wait_for(image_first_token.wait(), timeout=30)
        cancel_ns = time.monotonic_ns()
        image_task.cancel()
        try:
            await image_task
        except asyncio.CancelledError:
            pass

        await asyncio.wait_for(asyncio.gather(*text_tasks), timeout=40)
        text_summary = []
        text_client_rows: list[dict[str, object]] = []
        for lane, result in enumerate(text_results):
            assert result is not None
            rows = result.pop("events")
            for row in rows:
                row.update(episode_id=episode_id, arm="R", block=f"{args.cell}-block1", trial=0)
            text_client_rows.extend(rows)
            text_summary.append({
                "request_id": result.get("request_id"),
                "token_count": len(result.get("token_ids") or []),
                "error": result.get("error"),
                "token_ids": result.get("token_ids"),
            })
        append_jsonl(Path(args.client_events), text_client_rows)
        await pilot.flush_cuda_events(episode_id)
        result = {
            "status": "B_cancel_exercise_completed",
            "episode_id": episode_id,
            "anchor_ns": anchor_ns,
            "cancel_requested_ns": cancel_ns,
            "image_request_id": image_state.get("request_id"),
            "image_partial_tokens": len(image_state.get("token_ids", [])),
            "image_stream_cancelled": image_task.cancelled(),
            "text": text_summary,
            "image_sha256": pilot.image_hashes[0],
            "note": "Image SSE was deliberately disconnected after the first generated image token; expected partial image output is not a validation failure.",
        }
        Path(args.results_file).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "text"}, ensure_ascii=False), flush=True)
        return 0
    finally:
        await pilot.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
