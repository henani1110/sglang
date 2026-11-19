#!/usr/bin/env python3
"""Run one frozen A-stage configuration/probe cell."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

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
    parser.add_argument("--probe", type=int, choices=(0, 1), required=True)
    parser.add_argument("--text-tokens", type=int, default=192)
    parser.add_argument("--episodes-per-arm", type=int, default=8)
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

        for block, order in ((1, ("T", "R")), (2, ("R", "T"))):
            for arm in order:
                started = time.monotonic()
                for trial in range(args.episodes_per_arm):
                    row = await pilot.run_episode(
                        block=f"{args.cell}-probe{args.probe}-{block}",
                        arm=arm,
                        trial=trial,
                        image_index=trial % 4,
                    )
                    errors = pilot.validate_episode(
                        row,
                        expected_text_tokens=args.text_tokens,
                        require_image=(arm == "R"),
                    )
                    if errors:
                        await pilot.save_results("A_stopped_measurement_failure")
                        print(
                            json.dumps(
                                {
                                    "episode_id": row.get("episode_id"),
                                    "errors": errors,
                                    "elapsed_s": time.monotonic() - started,
                                }
                            ),
                            flush=True,
                        )
                        return 21
                    print(
                        f"cell={args.cell} probe={args.probe} block={block} "
                        f"arm={arm} trial={trial + 1}/{args.episodes_per_arm} "
                        f"elapsed_s={time.monotonic() - started:.1f}",
                        flush=True,
                    )

        if args.probe:
            diagnostic = await pilot.run_episode(
                block=f"{args.cell}-probe1-diagnostic",
                arm="R",
                trial=0,
                image_index=0,
                diagnostic=True,
                text_tokens=64,
            )
            errors = pilot.validate_episode(
                diagnostic, expected_text_tokens=64, require_image=True
            )
            if errors:
                await pilot.save_results("A_diagnostic_episode_failure")
                print(json.dumps({"diagnostic_errors": errors}), flush=True)
                return 22
            await pilot.flush_cuda_events(diagnostic["episode_id"])

        await pilot.flush_cuda_events(f"{args.cell}-probe{args.probe}-final")
        await pilot.save_results("A_completed")
        print(
            json.dumps(
                {
                    "status": "A_completed",
                    "cell": args.cell,
                    "probe": args.probe,
                    "measured_episodes": 4 * args.episodes_per_arm,
                    "blocks": 2,
                    "order": [["T", "R"], ["R", "T"]],
                    "result_file": args.results_file,
                }
            ),
            flush=True,
        )
        return 0
    finally:
        if not Path(args.results_file).exists():
            await pilot.save_results("A_interrupted")
        await pilot.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
