#!/usr/bin/env python3
"""Checkpointed, paired N/F/S/U main campaign for C; never drops failed seeds."""
from __future__ import annotations

import itertools
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

PROJECT = Path("/home/chx/project/vl-stage-split-20261106-191642")
SSD = Path("/media/chx/Win10/vl-stage-split-20261106-191642")
RUNNER = PROJECT / "continuation/scripts/run_c_cell.sh"
LOG_DIR = SSD / "continuation/logs"
PROGRESS = LOG_DIR / "C-main-progress.jsonl"
POLICY = {
    "N": {"prefill_interval": 1, "lm_budget": 2048, "encoder_budget": 0},
    "F": {"prefill_interval": 2, "lm_budget": 1024, "encoder_budget": 0},
    "S": {"prefill_interval": 1, "lm_budget": 2048, "encoder_budget": 0},
    "U": {"prefill_interval": 1, "lm_budget": 2048, "encoder_budget": 576},
}
SIZES = (512, 768)
STREAMS = (2, 6)
BLOCKS = (1, 2, 3)
TRIALS = tuple(range(12))


def append(row: dict[str, Any]) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()


def valid_row(row: dict[str, Any], streams: int) -> bool:
    text = row.get("text") or []
    image = row.get("image") or {}
    return (
        not row.get("error")
        and not (row.get("validation_errors") or [])
        and len(text) == streams
        and all(x.get("token_count") == 192 and not x.get("error") for x in text)
        and image.get("token_count") == 16
        and not image.get("error")
    )


def complete_trials(prefix: str, streams: int) -> set[int]:
    good: set[int] = set()
    for path in sorted((SSD / "results").glob(prefix + "-a*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for row in payload.get("episodes", []):
            if valid_row(row, streams):
                good.add(int(row.get("trial_seed", row.get("trial", -1))))
    return good


def next_attempt(prefix: str) -> int:
    values = []
    for path in (SSD / "results").glob(prefix + "-a*.json"):
        try:
            values.append(int(path.stem.rsplit("-a", 1)[1]))
        except ValueError:
            pass
    return max(values, default=-1) + 1


def tasks() -> list[tuple[str, int, int, int]]:
    """Balance policy position within each paired load/block, then interleave cells."""
    rows = []
    policies = ("N", "F", "S", "U")
    for load_index, (size, streams) in enumerate(itertools.product(SIZES, STREAMS)):
        for block in BLOCKS:
            rotation = (load_index + block - 1) % len(policies)
            order = list(policies[rotation:]) + list(policies[:rotation])
            if (load_index + block) % 2:
                order.reverse()
            rows.extend((policy, size, streams, block) for policy in order)
    random.Random(20261107).shuffle(rows)
    return rows


def run_cell(strategy: str, size: int, streams: int, block: int) -> None:
    prefix = f"Cmain-{strategy}-L{size}-D{streams}-B{block}"
    deferred_rounds = 0
    empty_retries = 0
    while complete_trials(prefix, streams) != set(TRIALS):
        good_before = complete_trials(prefix, streams)
        missing = sorted(set(TRIALS) - good_before)
        attempt = next_attempt(prefix)
        label = f"{prefix}-a{attempt}"
        cfg = POLICY[strategy]
        cmd = [
            "bash", str(RUNNER), label, strategy,
            "--phase", "measurement", "--block", f"C-block{block}",
            "--trial-ids", ",".join(str(x) for x in missing),
            "--attempt-id", str(attempt), "--episodes", str(len(missing)),
            "--text-streams", str(streams), "--image-size", str(size),
            "--max-seconds", "1800", "--prefill-interval", str(cfg["prefill_interval"]),
            "--lm-budget", str(cfg["lm_budget"]), "--encoder-budget", str(cfg["encoder_budget"]),
        ]
        started = time.time()
        rc = subprocess.run(cmd, check=False).returncode
        good_after = complete_trials(prefix, streams)
        new = sorted(good_after - good_before)
        append({
            "kind": "batch_finished", "label": label, "strategy": strategy,
            "load": {"image_size": size, "text_streams": streams}, "block": block,
            "trial_ids": missing, "new_valid_trials": new,
            "valid_trials_after_batch": sorted(good_after), "runner_exit": rc,
            "elapsed_s": round(time.time() - started, 2),
            "config": cfg,
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        if rc != 0:
            # A failed runner batch is retained in its logs. Give the external service
            # a short recovery interval, then retry only the missing paired seeds.
            time.sleep(min(60, 5 * (attempt + 1)))
        if good_after - good_before:
            empty_retries = 0
            deferred_rounds = 0
        else:
            empty_retries += 1
            deferred_rounds += 1
            if empty_retries >= 3:
                append({
                    "kind": "cell_deferred_for_diagnostics", "prefix": prefix,
                    "attempt": attempt, "missing": missing, "runner_exit": rc,
                    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
                return


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    jobs = tasks()
    append({
        "kind": "campaign_start_or_resume", "jobs": len(jobs),
        "policies": POLICY, "trial_ids": list(TRIALS),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    # Repeat full sweeps so a temporary failure in one cell never prevents the
    # remaining cells from progressing. Resume from valid per-seed checkpoints.
    sweep = 0
    while True:
        incomplete = []
        for strategy, size, streams, block in jobs:
            prefix = f"Cmain-{strategy}-L{size}-D{streams}-B{block}"
            if complete_trials(prefix, streams) == set(TRIALS):
                continue
            run_cell(strategy, size, streams, block)
            if complete_trials(prefix, streams) != set(TRIALS):
                incomplete.append(prefix)
        summary = {
            "kind": "sweep_finished", "sweep": sweep,
            "incomplete_cells": incomplete,
            "complete_cells": len(jobs) - len(incomplete),
            "total_cells": len(jobs),
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        append(summary)
        print(json.dumps(summary), flush=True)
        if not incomplete:
            break
        sweep += 1
        # Every revisit is preceded by diagnosis of the newest failure artifacts;
        # the coordinator will then resume only missing IDs after recovery.
        print(json.dumps({"kind": "incomplete_cells_saved_for_recovery", "cells": incomplete}), flush=True)
        time.sleep(min(300, 15 * (sweep + 1)))


if __name__ == "__main__":
    main()
