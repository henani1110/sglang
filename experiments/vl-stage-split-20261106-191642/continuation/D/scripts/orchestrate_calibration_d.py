#!/usr/bin/env python3
"""Checkpointed 64-run independent calibration driver for D."""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_ID = "vl-stage-split-20261106-191642"
D_S = Path(f"/media/chx/Win10/{RUN_ID}/continuation/D")
SCRIPT = Path(f"/home/chx/project/{RUN_ID}/continuation/D/scripts/run_d_cell.sh")
INDEX = D_S / "manifests" / "calibration-run-index.json"
LOG = D_S / "logs" / "calibration" / "orchestrator.jsonl"
VALIDATOR = Path(f"/home/chx/project/{RUN_ID}/continuation/D/scripts/validate_calibration_trace_d.py")
TRACE_AUDIT = D_S / "logs" / "calibration" / "trace-eligibility.jsonl"


def append(row: dict[str, Any]) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def completed_offsets(policy: str, grid: int, lanes: int, arrival: str) -> set[int]:
    if not INDEX.exists():
        return set()
    payload = json.loads(INDEX.read_text(encoding="utf-8"))
    result: set[int] = set()
    for row in payload.get("runs", []):
        if (
            row.get("policy") != policy
            or int(row.get("image_size", -1)) != grid
            or int(row.get("text_streams", -1)) != lanes
            or row.get("arrival_mode") != arrival
            or row.get("status") != "completed"
            or int(row.get("process_exit", 1)) != 0
            or row.get("analysis_eligible") is not True
        ):
            continue
        try:
            result_path = Path(row["result_path"])
            if json.loads(result_path.read_text()).get("status") == "completed":
                seed = int(row["seed"])
                if 300 <= seed <= 303:
                    result.add(seed - 300)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return result


def refresh_eligibility() -> None:
    completed = subprocess.run(
        [sys.executable, str(VALIDATOR), "--index", str(INDEX), "--audit-log", str(TRACE_AUDIT)],
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"calibration trace validation failed with exit {completed.returncode}; checkpoint retained")


def next_attempt(policy: str, grid: int, lanes: int, arrival: str) -> int:
    attempts = []
    if INDEX.exists():
        payload = json.loads(INDEX.read_text(encoding="utf-8"))
        for row in payload.get("runs", []):
            if (
                row.get("policy") == policy
                and int(row.get("image_size", -1)) == grid
                and int(row.get("text_streams", -1)) == lanes
                and row.get("arrival_mode") == arrival
            ):
                try:
                    attempts.append(int(str(row.get("attempt", "a0")).removeprefix("a")))
                except ValueError:
                    pass
    # A server can fail before the per-seed index is written. The orchestrator
    # log still records that attempt, so the next run must use a fresh ID and
    # preserve the original failure files instead of overwriting them.
    if LOG.exists():
        for line in LOG.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                row.get("event") == "cell_start"
                and row.get("policy") == policy
                and int(row.get("grid", -1)) == grid
                and int(row.get("lanes", -1)) == lanes
                and row.get("arrival") == arrival
            ):
                try:
                    attempts.append(int(row["attempt"]))
                except (KeyError, TypeError, ValueError):
                    pass
    return max(attempts, default=-1) + 1


def main() -> None:
    conditions = list(itertools.product((512, 768), (2, 6), ("uniform", "burst")))
    failures = 0
    refresh_eligibility()
    for condition_index, (grid, lanes, arrival) in enumerate(conditions):
        policy_order = ("N", "S") if condition_index % 2 == 0 else ("S", "N")
        for policy in policy_order:
            done = completed_offsets(policy, grid, lanes, arrival)
            pending = [offset for offset in range(4) if offset not in done]
            if not pending:
                append({"time_utc": datetime.now(timezone.utc).isoformat(), "event": "cell_already_complete", "grid": grid, "lanes": lanes, "arrival": arrival, "policy": policy})
                continue
            attempt = next_attempt(policy, grid, lanes, arrival)
            command = ["bash", str(SCRIPT), "calibration", policy, str(grid), str(lanes), arrival, "0", f"a{attempt}", *map(str, pending)]
            append({"time_utc": datetime.now(timezone.utc).isoformat(), "event": "cell_start", "grid": grid, "lanes": lanes, "arrival": arrival, "policy": policy, "attempt": attempt, "pending_offsets": pending})
            completed = subprocess.run(command, check=False)
            append({"time_utc": datetime.now(timezone.utc).isoformat(), "event": "cell_end", "grid": grid, "lanes": lanes, "arrival": arrival, "policy": policy, "attempt": attempt, "exit_code": completed.returncode})
            refresh_eligibility()
            if completed.returncode != 0:
                failures += 1
                print(f"calibration cell failed; trace eligibility was refreshed and evidence retained; diagnose before resuming: {command}", file=sys.stderr)
                raise SystemExit(completed.returncode)
            still_pending = [offset for offset in pending if offset not in completed_offsets(policy, grid, lanes, arrival)]
            if still_pending:
                append({"time_utc": datetime.now(timezone.utc).isoformat(), "event": "cell_trace_ineligible", "grid": grid, "lanes": lanes, "arrival": arrival, "policy": policy, "attempt": attempt, "pending_offsets": still_pending})
                print(f"cell output completed but required trace joins are invalid; evidence retained; inspect before retry: {command}", file=sys.stderr)
                raise SystemExit(20)

    indexed = json.loads(INDEX.read_text(encoding="utf-8")).get("runs", []) if INDEX.exists() else []
    missing = []
    for grid, lanes, arrival in conditions:
        for policy in ("N", "S"):
            if completed_offsets(policy, grid, lanes, arrival) != {0, 1, 2, 3}:
                missing.append({"grid": grid, "lanes": lanes, "arrival": arrival, "policy": policy})
    if missing:
        raise SystemExit(f"calibration matrix still has incomplete seed-policy cells: {missing}")
    print(json.dumps({"status": "calibration_runs_complete", "indexed_runs": len(indexed), "analysis_eligible_runs": sum(1 for row in indexed if row.get("analysis_eligible") is True), "failures": failures, "completed_seed_policy_cells": 16, "completed_runs": 64}))


if __name__ == "__main__":
    main()
