#!/usr/bin/env python3
"""Queue two full-resolution images concurrently to verify U's encoder cap."""
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
from trace_client import Pilot, append_jsonl


async def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "base-url", "model-path", "image-dir", "control-file", "client-events",
        "diagnostics-file", "equivalence-file", "results-file", "cell", "strategy",
        "phase", "block", "trial-ids", "attempt-id", "episodes", "text-streams",
        "trial-start", "text-tokens", "image-size", "arrival-offset-ms", "prefill-interval",
        "lm-budget", "encoder-budget",
    ):
        parser.add_argument(f"--{name}", required=name not in {
            "trial-ids", "trial-start", "attempt-id", "episodes", "arrival-offset-ms",
            "prefill-interval", "lm-budget", "encoder-budget",
        })
    args = parser.parse_args()
    streams = int(args.text_streams)
    text_tokens = int(args.text_tokens)
    pilot = Pilot(SimpleNamespace(
        base_url=args.base_url,
        client_events=args.client_events,
        control_file=args.control_file,
        image_dir=args.image_dir,
        model_path=args.model_path,
        model_name="qwen3vlpilot",
        text_tokens=text_tokens,
        text_streams=streams,
        arrival_offset_ms=float(args.arrival_offset_ms),
        diagnostics_file=args.diagnostics_file,
        equivalence_file=args.equivalence_file,
        results_file=args.results_file,
    ))
    episode_id = f"{args.block}-attempt{args.attempt_id}-{args.strategy}-two-image-probe"
    block_id = f"{args.block}-attempt{args.attempt_id}"
    try:
        health = await pilot.client.get(args.base_url.rstrip("/") + "/health")
        health.raise_for_status()
        pilot.set_mode("NORMAL", arm=args.strategy, episode_id=episode_id, image_index=None)
        await pilot.reset_caches()

        token_counts = [0] * streams
        ready_events = [asyncio.Event() for _ in range(streams)]
        trigger_event = asyncio.Event()
        trigger_data: dict[str, object] = {}
        text_results: list[dict[str, object] | None] = [None] * streams

        async def run_text(lane: int) -> None:
            label = f"{episode_id}-text{lane}"
            payload = {
                "model": pilot.args.model_name,
                "prompt": pilot.prompts[lane],
                "max_tokens": text_tokens,
                "temperature": 0,
                "ignore_eos": True,
                "stream": True,
                "return_token_ids": True,
                "user": label,
            }

            def on_token(count: int, time_ns: int) -> bool:
                token_counts[lane] += count
                if not trigger_event.is_set() and token_counts[0] >= 16 and all(token_counts[1:]):
                    trigger_data["time_ns"] = time_ns
                    trigger_data["counts"] = list(token_counts)
                    trigger_event.set()
                    return True
                return False

            text_results[lane] = await pilot.read_stream(
                "/v1/completions", payload, request_label=label,
                lane=lane, ready=ready_events[lane], on_token=on_token,
            )
            return text_results[lane]

        text_tasks = [asyncio.create_task(run_text(lane)) for lane in range(streams)]
        await asyncio.wait_for(trigger_event.wait(), timeout=30)
        trigger_ns = int(trigger_data["time_ns"])
        target_ns = trigger_ns + int(float(args.arrival_offset_ms) * 1e6)
        if target_ns > time.monotonic_ns():
            await asyncio.sleep((target_ns - time.monotonic_ns()) / 1e9)
        anchor_ns = time.monotonic_ns()
        active = [not task.done() for task in text_tasks]
        image_send_ns: list[int | None] = [None, None]
        image_tasks = []
        for image_index in (0, 1):
            def on_send(time_ns: int, index: int = image_index) -> None:
                image_send_ns[index] = time_ns

            image_tasks.append(asyncio.create_task(pilot.request_image(
                image_index,
                episode_id=f"{episode_id}-image{image_index}",
                on_send=on_send,
            )))

        gathered = await asyncio.wait_for(
            asyncio.gather(*text_tasks, *image_tasks, return_exceptions=True), timeout=40
        )
        text_rows = gathered[:streams]
        image_rows = gathered[streams:]
        client_rows = [{
            "kind": "episode_anchor", "episode_id": episode_id, "block": block_id,
            "pair_block": args.block, "arm": args.strategy, "time_ns": anchor_ns,
            "trigger_ns": trigger_ns, "arrival_offset_ms": (anchor_ns-trigger_ns)/1e6,
            "text_token_counts_at_trigger": trigger_data["counts"],
            "text_token_counts_at_anchor": token_counts,
            "active_text_streams_at_anchor": active,
            "image_indices": [0, 1],
        }]
        normalized_text = []
        for lane, result in enumerate(text_rows):
            if isinstance(result, Exception) or result is None:
                normalized_text.append({"lane": lane, "error": str(result), "token_count": 0, "token_ids": []})
                continue
            rows = result.pop("events")
            for event in rows:
                event.update(episode_id=episode_id, block=block_id, pair_block=args.block, arm=args.strategy, trial=0)
            client_rows.extend(rows)
            normalized_text.append({
                "lane": lane,
                "request_id": result.get("request_id"),
                "token_count": len(result.get("token_ids") or []),
                "error": result.get("error"),
                "prompt_token_ids": result.get("prompt_token_ids"),
                "token_ids": result.get("token_ids") or [],
            })
        normalized_images = []
        for image_index, result in enumerate(image_rows):
            if isinstance(result, Exception) or result is None:
                normalized_images.append({"image_index": image_index, "error": str(result), "token_count": 0, "token_ids": []})
                continue
            for event in result.pop("events"):
                event.update(episode_id=episode_id, block=block_id, pair_block=args.block, arm=args.strategy, trial=0)
                client_rows.append(event)
            normalized_images.append({
                key: result.get(key) for key in ("request_id", "started_ns", "finished_ns", "image_index", "image_sha256", "error")
            } | {"token_count": len(result.get("token_ids") or []), "token_ids": result.get("token_ids") or []})
            client_rows.append({
                "kind": "image_request_send", "episode_id": episode_id, "block": block_id,
                "pair_block": args.block, "arm": args.strategy, "image_index": image_index,
                "time_ns": image_send_ns[image_index], "request_id": result.get("request_id"),
            })
        append_jsonl(Path(args.client_events), client_rows)
        errors = []
        if not all(active): errors.append("not all text streams active at dual-image anchor")
        if any(len(item.get("token_ids") or []) != text_tokens or item.get("error") for item in normalized_text):
            errors.append("one or more text streams failed or returned the wrong token count")
        if any(len(item.get("token_ids") or []) != 16 or item.get("error") for item in normalized_images):
            errors.append("one or more image streams failed or returned the wrong token count")
        row = {
            "episode_id": episode_id,
            "block": block_id,
            "pair_block": args.block,
            "arm": args.strategy,
            "phase": "probe",
            "image_size": int(args.image_size),
            "text_streams": streams,
            "text_output_token_target": text_tokens,
            "image_indices": [0, 1],
            "anchor_ns": anchor_ns,
            "image_send_ns": image_send_ns,
            "active_text_streams_at_anchor": active,
            "text_token_counts_at_trigger": trigger_data["counts"],
            "text": normalized_text,
            "images": normalized_images,
            "validation_errors": errors,
            "strategy_config": {
                "strategy": args.strategy, "lm_budget": int(args.lm_budget),
                "encoder_budget": int(args.encoder_budget),
                "prefill_interval": int(args.prefill_interval),
            },
        }
        pilot.results.append(row)
        try:
            await pilot.flush_cuda_events(episode_id)
        except Exception as exc:
            append_jsonl(Path(args.diagnostics_file), [{
                "kind": "cuda_event_flush_error", "episode_id": episode_id,
                "error": f"{type(exc).__name__}: {exc}", "time_ns": time.monotonic_ns(),
            }])
        status = f"C_{args.strategy}_two_image_probe_completed" if not errors else f"C_{args.strategy}_two_image_probe_invalid"
        await pilot.save_results(status)
        print(json.dumps({
            "status": status,
            "errors": errors,
            "text_tokens": [x["token_count"] for x in normalized_text],
            "image_tokens": [x["token_count"] for x in normalized_images],
            "image_request_ids": [x.get("request_id") for x in normalized_images],
            "image_send_ns": image_send_ns,
        }, ensure_ascii=False), flush=True)
        return 0
    finally:
        await pilot.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
