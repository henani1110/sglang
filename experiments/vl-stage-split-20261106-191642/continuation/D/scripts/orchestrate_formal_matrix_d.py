#!/usr/bin/env python3
"""Pre-register and checkpoint the complete 432-run D holdout matrix."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_ID = "vl-stage-split-20261106-191642"
D_P = Path(f"/home/chx/project/{RUN_ID}/continuation/D")
D_S = Path(f"/media/chx/Win10/{RUN_ID}/continuation/D")
SCRIPT = D_P / "scripts" / "run_d_cell.sh"
MANIFEST = D_S / "manifests" / "d-run-manifest.json"
PROFILE = D_S / "manifests" / "estimator-profile.json"
DATASET = D_S / "datasets" / "holdout-final" / "holdout-selected-manifest.json"
SCHEDULE = D_S / "manifests" / "formal-pre-registered-schedule.json"
LOG = D_S / "logs" / "formal" / "orchestrator.jsonl"
LOCK = D_S / "manifests" / "formal-orchestrator.lock"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def append(row: dict[str, Any]) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def rotate(values: list[str], amount: int) -> list[str]:
    amount %= len(values)
    return values[amount:] + values[:amount]


def build_schedule(dataset_sha: str, profile_sha: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    condition_index = 0
    cell_order = 0
    for grid, lanes, arrival in itertools.product((512, 768), (2, 6), ("uniform", "burst")):
        policies = ["N", "F", "S", "A"] + (["U"] if arrival == "burst" else [])
        base_order = rotate(policies, condition_index % len(policies))
        second_order = rotate(base_order, (len(policies) + 1) // 2)
        block_orders = (base_order, second_order, list(reversed(base_order)))
        for block, order in enumerate(block_orders):
            for position, policy in enumerate(order):
                cell_order += 1
                cell = {
                    "cell_order": cell_order,
                    "condition_index": condition_index,
                    "grid": grid,
                    "lanes": lanes,
                    "arrival": arrival,
                    "policy": policy,
                    "block": block,
                    "policy_position": position,
                    "block_policy_order": order,
                    "seed_range": [1000 + block * 4, 1003 + block * 4],
                }
                cells.append(cell)
                for offset in range(4):
                    seed = 1000 + block * 4 + offset
                    rows.append(
                        {
                            **cell,
                            "seed": seed,
                            "seed_offset": offset,
                            "pair_key": f"g{grid}-l{lanes}-{arrival}-b{block}-s{seed}",
                        }
                    )
        condition_index += 1
    if len(rows) != 432 or len(cells) != 108:
        raise RuntimeError(f"schedule construction mismatch: {len(rows)} runs / {len(cells)} cells")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "pre_registered",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": RUN_ID,
        "plan_revision": "v4.3",
        "dataset_manifest_sha256": dataset_sha,
        "estimator_profile_sha256": profile_sha,
        "planned_policy_cells": 36,
        "planned_block_cells": 108,
        "planned_seed_policy_runs": 432,
        "seeds": list(range(1000, 1012)),
        "rows": rows,
    }
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    payload["schedule_sha256"] = digest
    payload["policy_cell_order"] = cells
    return payload


def verify_or_create_schedule(dataset_sha: str, profile_sha: str) -> dict[str, Any]:
    if SCHEDULE.exists():
        saved = json.loads(SCHEDULE.read_text(encoding="utf-8"))
        saved_hash = saved.pop("schedule_sha256", None)
        saved_cells = saved.pop("policy_cell_order", None)
        digest = hashlib.sha256(json.dumps(saved, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        saved["schedule_sha256"] = saved_hash
        if saved_hash != digest:
            raise RuntimeError("saved formal schedule hash is invalid; preserve it and inspect before proceeding")
        if saved.get("dataset_manifest_sha256") != dataset_sha or saved.get("estimator_profile_sha256") != profile_sha:
            raise RuntimeError("formal schedule inputs differ from the frozen dataset/profile; do not overwrite the pre-registration")
        if saved.get("planned_seed_policy_runs") != 432 or len(saved.get("rows", [])) != 432:
            raise RuntimeError("saved formal schedule does not contain all 432 planned runs")
        saved["policy_cell_order"] = saved_cells
        return saved
    schedule = build_schedule(dataset_sha, profile_sha)
    atomic_json(SCHEDULE, schedule)
    return schedule


def successful_keys(payload: dict[str, Any]) -> set[tuple[int, int, str, str, int]]:
    result: set[tuple[int, int, str, str, int]] = set()
    for row in payload.get("matrix", {}).get("seed_records", []):
        if row.get("status") != "completed" or int(row.get("process_exit", 1)) != 0:
            continue
        try:
            result_path = Path(row["result_path"])
            if json.loads(result_path.read_text(encoding="utf-8")).get("status") != "completed":
                continue
            result.add(
                (
                    int(row["image_size"]),
                    int(row["text_streams"]),
                    str(row["arrival_mode"]),
                    str(row["policy"]),
                    int(row["seed"]),
                )
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return result


def next_attempt(payload: dict[str, Any], cell: dict[str, Any]) -> int:
    attempts: list[int] = []
    for row in payload.get("matrix", {}).get("seed_records", []):
        if (
            int(row.get("image_size", -1)) == cell["grid"]
            and int(row.get("text_streams", -1)) == cell["lanes"]
            and row.get("arrival_mode") == cell["arrival"]
            and row.get("policy") == cell["policy"]
            and int(row.get("block", -1)) == cell["block"] + 1
        ):
            try:
                attempts.append(int(str(row.get("attempt", "a0")).removeprefix("a")))
            except ValueError:
                pass
    if LOG.exists():
        for line in LOG.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                row.get("event") == "cell_start"
                and int(row.get("grid", -1)) == cell["grid"]
                and int(row.get("lanes", -1)) == cell["lanes"]
                and row.get("arrival") == cell["arrival"]
                and row.get("policy") == cell["policy"]
                and int(row.get("block", -1)) == cell["block"]
            ):
                try:
                    attempts.append(int(row["attempt"]))
                except (KeyError, TypeError, ValueError):
                    pass
    return max(attempts, default=-1) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true", help="write/verify the frozen 432-run schedule without starting GPU jobs")
    args = parser.parse_args()

    for path in (MANIFEST, PROFILE, DATASET, SCRIPT):
        if not path.exists():
            raise SystemExit(f"required D artifact is missing: {path}")
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    if profile.get("frozen") is not True or set(profile.get("cells", {})) != {"512:2", "512:6", "768:2", "768:6"}:
        raise SystemExit("estimator profile is not frozen with all four registered cells")
    profile_sha = sha256(PROFILE)
    sidecar = PROFILE.with_suffix(PROFILE.suffix + ".sha256")
    if not sidecar.exists() or sidecar.read_text(encoding="utf-8").split()[0] != profile_sha:
        raise SystemExit("frozen estimator profile SHA-256 sidecar does not match")
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    if dataset.get("status") != "complete" or int(dataset.get("selected_count", 0)) != 32 or len(dataset.get("images", [])) != 32:
        raise SystemExit("formal holdout manifest is not the complete 32-image dataset")
    dataset_sha = sha256(DATASET)

    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another formal-matrix orchestrator holds the D lock") from exc
        schedule = verify_or_create_schedule(dataset_sha, profile_sha)
        print(
            json.dumps(
                {
                    "status": "schedule_pre_registered",
                    "policy_cells": schedule["planned_policy_cells"],
                    "seed_policy_runs": schedule["planned_seed_policy_runs"],
                    "schedule_sha256": schedule["schedule_sha256"],
                    "path": str(SCHEDULE),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if args.plan_only:
            return

        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        done = successful_keys(manifest)
        cells_by_order: dict[int, dict[str, Any]] = {}
        for row in schedule["rows"]:
            cells_by_order.setdefault(int(row["cell_order"]), row)

        for cell_order in sorted(cells_by_order):
            cell = cells_by_order[cell_order]
            expected_seeds = [1000 + cell["block"] * 4 + offset for offset in range(4)]
            pending = [seed - (1000 + cell["block"] * 4) for seed in expected_seeds if (cell["grid"], cell["lanes"], cell["arrival"], cell["policy"], seed) not in done]
            if not pending:
                append({"time_utc": datetime.now(timezone.utc).isoformat(), "event": "cell_already_complete", **cell})
                continue
            attempt = next_attempt(manifest, cell)
            command = [
                "bash", str(SCRIPT), "formal", str(cell["policy"]), str(cell["grid"]),
                str(cell["lanes"]), str(cell["arrival"]), str(cell["block"]), f"a{attempt}",
                *map(str, pending),
            ]
            append(
                {
                    "time_utc": datetime.now(timezone.utc).isoformat(),
                    "event": "cell_start",
                    "attempt": attempt,
                    "pending_offsets": pending,
                    "command": command,
                    **cell,
                }
            )
            print(json.dumps({"status": "cell_start", "cell_order": cell_order, "policy": cell["policy"], "grid": cell["grid"], "lanes": cell["lanes"], "arrival": cell["arrival"], "block": cell["block"], "pending_offsets": pending, "attempt": attempt}, ensure_ascii=False), flush=True)
            completed = subprocess.run(command, check=False)
            append(
                {
                    "time_utc": datetime.now(timezone.utc).isoformat(),
                    "event": "cell_end",
                    "attempt": attempt,
                    "exit_code": completed.returncode,
                    **cell,
                }
            )
            if completed.returncode != 0:
                print(f"formal cell failed with exit {completed.returncode}; evidence retained; inspect and repair before resuming: {command}", file=sys.stderr, flush=True)
                raise SystemExit(completed.returncode)
            manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
            done = successful_keys(manifest)

        expected = {
            (int(row["grid"]), int(row["lanes"]), str(row["arrival"]), str(row["policy"]), int(row["seed"]))
            for row in schedule["rows"]
        }
        missing = sorted(expected - done)
        if missing:
            raise SystemExit(f"formal schedule is incomplete after this pass: {len(missing)} seed-policy runs remain; first missing keys={missing[:12]}")
        print(json.dumps({"status": "formal_matrix_complete", "completed_seed_policy_runs": len(done), "schedule_sha256": schedule["schedule_sha256"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
