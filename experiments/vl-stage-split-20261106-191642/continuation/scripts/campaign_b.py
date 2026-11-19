#!/usr/bin/env python3
"""Run one or more fixed image-plus-four-text B semantic episodes."""
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
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--text-tokens", type=int, default=192)
    args = parser.parse_args()

    pilot_args = SimpleNamespace(
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
    )
    pilot = Pilot(pilot_args)
    try:
        health = await pilot.client.get(args.base_url.rstrip("/") + "/health")
        health.raise_for_status()
        for trial in range(args.episodes):
            started = time.monotonic()
            row = await pilot.run_episode(
                block=f"{args.cell}-block1",
                arm="R",
                trial=trial,
                image_index=0,
            )
            errors = pilot.validate_episode(
                row, expected_text_tokens=args.text_tokens, require_image=True
            )
            if errors:
                await pilot.save_results("B_measurement_failure")
                print(json.dumps({"episode_id": row.get("episode_id"), "errors": errors}), flush=True)
                return 21
            print(
                f"cell={args.cell} trial={trial + 1}/{args.episodes} "
                f"elapsed_s={time.monotonic() - started:.1f}",
                flush=True,
            )
            await pilot.flush_cuda_events(row["episode_id"])
        await pilot.save_results("B_completed")
        print(
            json.dumps({
                "status": "B_completed",
                "cell": args.cell,
                "episodes": args.episodes,
                "result_file": args.results_file,
            }),
            flush=True,
        )
        return 0
    finally:
        await pilot.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
