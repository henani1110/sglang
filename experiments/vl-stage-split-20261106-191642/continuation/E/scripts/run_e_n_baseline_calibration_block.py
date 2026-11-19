#!/usr/bin/env python3
"""Run a balanced, seed-addressable E/N synthetic baseline replay block."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


RUN_ID = "vl-stage-split-20261106-191642"
DEFAULT_PROJECT = Path(f"/home/chx/project/{RUN_ID}/continuation/E")
DEFAULT_SSD = Path(f"/media/chx/Win10/{RUN_ID}/continuation/E")
DEFAULT_PYTHON = Path("/home/chx/project/vl-encode-pilot-20261106-085428/env/serve311/bin/python")
MODEL = Path("/media/chx/Win10/vl-encode-pilot-20261106-085428/models/Qwen3-VL-2B-Instruct")
CELLS = [(size, lanes, pattern) for size in (512, 768) for lanes in (2, 6) for pattern in ("uniform", "burst")]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def healthy(base_url: str) -> bool:
    try:
        with urlopen(base_url.rstrip("/") + "/health", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def run_one(
    *, python: Path, replay: Path, output_root: Path, asset_root: Path,
    base_url: str, seed: int, size: int, lanes: int, pattern: str,
    image_count: int, text_tokens: int, image_tokens: int, warmup: bool,
) -> dict:
    label = f"s{seed}-{size}-{lanes}lane-{pattern}"
    if warmup:
        label = f"warmup-{label}"
    output = output_root / ("warmup" if warmup else f"seed-{seed}") / f"{label}.json"
    assets = asset_root / ("warmup" if warmup else f"seed-{seed}") / f"{size}-{lanes}lane-{pattern}"
    log_path = output.with_suffix(".stdout.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python), str(replay), "--model", str(MODEL), "--base-url", base_url,
        "--output-json", str(output), "--asset-dir", str(assets),
        "--text-lanes", str(lanes), "--image-size", str(size),
        "--text-max-tokens", str(text_tokens), "--image-count", str(image_count),
        "--arrival-pattern", pattern, "--image-max-tokens", str(image_tokens),
        "--first-token-timeout-seconds", "120", "--seed", str(seed),
    ]
    started = datetime.now(timezone.utc).isoformat()
    with log_path.open("w", encoding="utf-8") as stream:
        proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    compact = None
    if proc.returncode == 0:
        try:
            lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            compact = json.loads(lines[-1]) if lines else None
        except Exception as error:
            compact = {"parse_error": str(error)}
    return {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "image_size": size,
        "text_lanes": lanes,
        "arrival_pattern": pattern,
        "image_count": image_count,
        "text_max_tokens": text_tokens,
        "image_max_tokens": image_tokens,
        "warmup_excluded": warmup,
        "command": command,
        "returncode": proc.returncode,
        "output_json": str(output),
        "asset_dir": str(assets),
        "stdout_log": str(log_path),
        "compact": compact,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--ssd", type=Path, default=DEFAULT_SSD)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--base-url", default="http://127.0.0.1:8285")
    parser.add_argument("--label", default="e-n-baseline-cal-v0.1")
    parser.add_argument("--block-id", help="Unique immutable output suffix; defaults to current UTC timestamp.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20261110, 20261122)))
    parser.add_argument("--image-count", type=int, default=12)
    parser.add_argument("--text-max-tokens", type=int, default=512)
    parser.add_argument("--image-max-tokens", type=int, default=16)
    parser.add_argument("--health-timeout-seconds", type=int, default=300)
    args = parser.parse_args()

    block_id = args.block_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_label = f"{args.label}-{block_id}"

    replay = args.ssd / "scripts/replay_live_joint_controller_e.py"
    server_script = args.project / "scripts/run_server_e.sh"
    output_root = args.ssd / "runs/n-baseline-calibration-v0.1" / block_id
    asset_root = args.ssd / "runs/n-baseline-calibration-assets-v0.1" / block_id
    log_root = args.ssd / "logs/n-baseline-calibration-v0.1" / block_id
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    config = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "block_id": block_id,
        "block_label": run_label,
        "arm": "E-N instrumented keep-all baseline; not formal",
        "gpu_selection": "run_server_e.sh pins GPU0 UUID",
        "model": str(MODEL),
        "vllm_mode": "N",
        "seeds": args.seeds,
        "cells": CELLS,
        "order": "forward on even seed index, reverse on odd seed index",
        "image_count_per_cell": args.image_count,
        "text_max_tokens_per_lane": args.text_max_tokens,
        "image_max_tokens": args.image_max_tokens,
        "warmup": {"cell": [512, 2, "uniform"], "images": 4, "text_max_tokens": 128, "excluded": True},
        "limitations": [
            "Synthetic charts and fixed output caps are engineering calibration only.",
            "Instrumented E/N keep-all replay is not a pristine uninstrumented vLLM baseline.",
            "No answer-quality, pruning, formal SLO, treatment comparison, or 480-run count is implied.",
        ],
    }
    (output_root / "block-config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    index_path = output_root / "run-index.jsonl"
    server_log = log_root / "server.log"
    env = os.environ.copy()
    env.update({"VL_PILOT_LABEL": run_label, "VL_STAGE_SPLIT_MODE": "N"})

    server = None
    exit_code = 0
    with server_log.open("w", encoding="utf-8") as server_stream:
        try:
            server = subprocess.Popen(["bash", str(server_script)], cwd=str(args.project), env=env,
                                      stdout=server_stream, stderr=subprocess.STDOUT, start_new_session=True)
            (log_root / "server-process.json").write_text(json.dumps({
                "pid": server.pid, "started_utc": datetime.now(timezone.utc).isoformat(),
                "label": run_label, "mode": "N", "server_script": str(server_script),
            }, indent=2) + "\n", encoding="utf-8")
            deadline = time.monotonic() + args.health_timeout_seconds
            while time.monotonic() < deadline and server.poll() is None and not healthy(args.base_url):
                time.sleep(2)
            if server.poll() is not None or not healthy(args.base_url):
                raise RuntimeError(f"server failed readiness; returncode={server.poll()}; inspect {server_log}")

            warm = run_one(
                python=args.python, replay=replay, output_root=output_root, asset_root=asset_root,
                base_url=args.base_url, seed=20261109, size=512, lanes=2, pattern="uniform",
                image_count=4, text_tokens=128, image_tokens=args.image_max_tokens, warmup=True,
            )
            append_jsonl(index_path, warm)
            if warm["returncode"] != 0 or not healthy(args.base_url):
                append_jsonl(index_path, {"status": "service_failed_after_warmup", "warmup": warm})
                raise RuntimeError("warmup failed or E/N service became unhealthy")

            total = len(args.seeds) * len(CELLS)
            done = 0
            for seed_index, seed in enumerate(args.seeds):
                ordered_cells = CELLS if seed_index % 2 == 0 else list(reversed(CELLS))
                for size, lanes, pattern in ordered_cells:
                    if server.poll() is not None or not healthy(args.base_url):
                        append_jsonl(index_path, {"status": "service_unhealthy", "seed": seed,
                                                  "cell": [size, lanes, pattern], "completed": done, "planned": total})
                        raise RuntimeError("E/N service exited or failed health check; block checkpoint preserved")
                    row = run_one(
                        python=args.python, replay=replay, output_root=output_root, asset_root=asset_root,
                        base_url=args.base_url, seed=seed, size=size, lanes=lanes, pattern=pattern,
                        image_count=args.image_count, text_tokens=args.text_max_tokens,
                        image_tokens=args.image_max_tokens, warmup=False,
                    )
                    append_jsonl(index_path, row)
                    done += 1
                    print(json.dumps({"done": done, "planned": total, "seed": seed,
                                      "cell": [size, lanes, pattern], "returncode": row["returncode"],
                                      "compact": row["compact"]}, ensure_ascii=False), flush=True)
                    if row["returncode"] != 0:
                        append_jsonl(output_root / "failures.jsonl", row)
                    if not healthy(args.base_url):
                        append_jsonl(index_path, {"status": "service_unhealthy_after_cell", "seed": seed,
                                                  "cell": [size, lanes, pattern], "completed": done, "planned": total})
                        raise RuntimeError("E/N service failed after cell; block checkpoint preserved")
        except Exception as error:
            exit_code = 1
            append_jsonl(index_path, {"status": "block_interrupted", "time_utc": datetime.now(timezone.utc).isoformat(),
                                      "error": repr(error), "traceback": traceback.format_exc()})
            print(f"BLOCK_ERROR {error!r}", file=sys.stderr, flush=True)
        finally:
            if server is not None and server.poll() is None:
                try:
                    os.killpg(server.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    server.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    exit_code = 2
                    append_jsonl(index_path, {"status": "server_did_not_drain_within_60s", "pid": server.pid,
                                              "action": "left running for diagnosis; no force kill issued"})

    final = {"finished_utc": datetime.now(timezone.utc).isoformat(), "returncode": exit_code,
             "server_returncode": server.poll() if server else None, "run_index": str(index_path),
             "server_log": str(server_log)}
    (output_root / "block-completion.json").write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
