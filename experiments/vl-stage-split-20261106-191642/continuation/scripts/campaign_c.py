#!/usr/bin/env python3
"""Run checkpointed, paired C-stage single-image episodes for one fixed policy."""
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
from trace_client import Pilot


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
    parser.add_argument("--strategy", choices=("N", "F", "S", "U"), required=True)
    parser.add_argument("--phase", choices=("calibration", "measurement", "probe"), required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--trial-ids", help="comma-separated paired trial IDs for exact retries")
    parser.add_argument("--attempt-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--text-streams", type=int, choices=(2, 6), required=True)
    parser.add_argument("--text-tokens", type=int, default=192)
    parser.add_argument("--image-size", type=int, choices=(512, 768), required=True)
    parser.add_argument("--arrival-offset-ms", type=float, default=5.0)
    parser.add_argument("--prefill-interval", type=int, default=1)
    parser.add_argument("--lm-budget", type=int, default=2048)
    parser.add_argument("--encoder-budget", type=int, default=0)
    args = parser.parse_args()

    pilot_args = SimpleNamespace(
        base_url=args.base_url,
        client_events=args.client_events,
        control_file=args.control_file,
        image_dir=args.image_dir,
        model_path=args.model_path,
        model_name="qwen3vlpilot",
        text_tokens=args.text_tokens,
        text_streams=args.text_streams,
        arrival_offset_ms=args.arrival_offset_ms,
        diagnostics_file=args.diagnostics_file,
        equivalence_file=args.equivalence_file,
        results_file=args.results_file,
    )
    pilot = Pilot(pilot_args)
    config = {
        "strategy": args.strategy,
        "phase": args.phase,
        "block": args.block,
        "attempt_id": args.attempt_id,
        "text_streams": args.text_streams,
        "image_size": args.image_size,
        "text_tokens": args.text_tokens,
        "image_tokens": 16,
        "arrival_offset_ms": args.arrival_offset_ms,
        "prefill_interval": args.prefill_interval,
        "lm_budget": args.lm_budget,
        "encoder_budget": args.encoder_budget,
        "trial_start": args.trial_start,
        "episode_count": len([x for x in args.trial_ids.split(",") if x.strip()]) if args.trial_ids else args.episodes,
    }
    failures = 0
    trial_ids = (
        [int(item.strip()) for item in args.trial_ids.split(",") if item.strip()]
        if args.trial_ids else list(range(args.trial_start, args.trial_start + args.episodes))
    )
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError(f"duplicate trial ids: {trial_ids}")
    try:
        health = await pilot.client.get(args.base_url.rstrip("/") + "/health")
        health.raise_for_status()
        for trial in trial_ids:
            unique_block = f"{args.block}-attempt{args.attempt_id}"
            started = time.monotonic()
            row = await pilot.run_episode(
                block=unique_block,
                arm=args.strategy,
                trial=trial,
                image_index=trial % 4,
            )
            errors = pilot.validate_episode(
                row, expected_text_tokens=args.text_tokens, require_image=True
            )
            row["pair_block"] = args.block
            row["attempt_id"] = args.attempt_id
            row["trial_seed"] = trial
            row["text_streams"] = args.text_streams
            row["image_size"] = args.image_size
            row["phase"] = args.phase
            row["strategy_config"] = config.copy()
            if errors:
                failures += 1
            await pilot.save_results("C_running")
            print(json.dumps({
                "cell": args.cell,
                "strategy": args.strategy,
                "block": args.block,
                "trial_seed": trial,
                "valid": not errors,
                "errors": errors,
                "elapsed_s": round(time.monotonic() - started, 3),
            }, ensure_ascii=False), flush=True)
        try:
            await pilot.flush_cuda_events(f"{args.cell}-{args.phase}-batch-final")
        except Exception as exc:
            from trace_client import append_jsonl

            append_jsonl(Path(args.diagnostics_file), [{
                "kind": "cuda_event_flush_error",
                "cell": args.cell,
                "error": f"{type(exc).__name__}: {exc}",
                "time_ns": time.monotonic_ns(),
            }])
        status = "C_completed" if failures == 0 else "C_completed_with_invalid_samples"
        await pilot.save_results(status)
        print(json.dumps({
            "status": status,
            "cell": args.cell,
            "phase": args.phase,
            "episodes": len(trial_ids),
            "valid": len(trial_ids) - failures,
            "invalid": failures,
            "result_file": args.results_file,
        }, ensure_ascii=False), flush=True)
        return 0
    finally:
        await pilot.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
