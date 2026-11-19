#!/usr/bin/env python3
"""Run and resume the independent F-profile calibration grid on the remote SSD."""
from __future__ import annotations

import itertools
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT = Path("/home/chx/project/vl-stage-split-20261106-191642")
SSD = Path("/media/chx/Win10/vl-stage-split-20261106-191642")
RUNNER = PROJECT / "continuation/scripts/run_c_cell.sh"
LOG_DIR = SSD / "continuation/logs"
OUT = SSD / "continuation/manifests/C-F-selected.json"
PROGRESS = LOG_DIR / "C-calibration-progress.jsonl"
SIZES = (512, 768)
STREAMS = (2, 6)
INTERVALS = (1, 2, 3)
LM_BUDGETS = (512, 1024, 2048)
TRIAL_IDS = (0, 1, 2)


def append_progress(row: dict[str, Any]) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def row_is_valid(row: dict[str, Any], streams: int) -> bool:
    texts = row.get("text") or []
    image = row.get("image") or {}
    return (
        not row.get("error")
        and not (row.get("validation_errors") or [])
        and len(texts) == streams
        and all(x.get("token_count") == 192 and not x.get("error") for x in texts)
        and image.get("token_count") == 16
        and not image.get("error")
    )


def valid_trials(prefix: str, streams: int) -> set[int]:
    found: set[int] = set()
    for path in sorted((SSD / "results").glob(prefix + "-a*.json")):
        try:
            result = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for row in result.get("episodes", []):
            if row_is_valid(row, streams):
                found.add(int(row.get("trial_seed", row.get("trial", -1))))
    return found


def next_attempt(prefix: str) -> int:
    attempts = []
    for path in (SSD / "results").glob(prefix + "-a*.json"):
        try:
            attempts.append(int(path.stem.rsplit("-a", 1)[1]))
        except ValueError:
            pass
    return max(attempts, default=-1) + 1


def run_group(interval: int, budget: int, size: int, streams: int) -> None:
    prefix = f"Ccal-I{interval}-Q{budget}-L{size}-D{streams}"
    block = f"C-cal-L{size}-D{streams}"
    consecutive_empty_attempts = 0
    while valid_trials(prefix, streams) != set(TRIAL_IDS):
        missing = sorted(set(TRIAL_IDS) - valid_trials(prefix, streams))
        attempt = next_attempt(prefix)
        label = f"{prefix}-a{attempt}"
        cmd = [
            "bash", str(RUNNER), label, "F",
            "--phase", "calibration", "--block", block,
            "--trial-ids", ",".join(str(x) for x in missing),
            "--attempt-id", str(attempt), "--episodes", str(len(missing)),
            "--text-streams", str(streams), "--image-size", str(size),
            "--max-seconds", "1800", "--prefill-interval", str(interval),
            "--lm-budget", str(budget), "--encoder-budget", "0",
        ]
        started = time.time()
        rc = subprocess.run(cmd, check=False).returncode
        good = valid_trials(prefix, streams)
        append_progress({
            "kind": "batch_finished",
            "label": label,
            "strategy": "F",
            "profile": {"interval": interval, "lm_budget": budget},
            "load": {"image_size": size, "text_streams": streams},
            "trial_ids": missing,
            "valid_trials_after_batch": sorted(good),
            "runner_exit": rc,
            "elapsed_s": round(time.time() - started, 2),
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        if rc != 0:
            raise RuntimeError(f"batch {label} runner exited {rc}; inspect {PROJECT / 'logs' / (label + '-server.log')}")
        if not (set(TRIAL_IDS) - good):
            consecutive_empty_attempts = 0
        else:
            consecutive_empty_attempts += 1
            if consecutive_empty_attempts >= 20:
                raise RuntimeError(f"calibration {prefix} still has invalid/missing seeds after 20 retries")


def percentile(values: list[float], p: float = 0.95) -> float | None:
    if not values:
        return None
    items = sorted(values)
    x = (len(items) - 1) * p
    lo = int(x)
    hi = min(lo + 1, len(items) - 1)
    return items[lo] + (items[hi] - items[lo]) * (x - lo)


def episode_metrics(row: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any] | None:
    anchor = int(row.get("anchor_ns", 0))
    episode_id = row.get("episode_id")
    if not anchor or not episode_id:
        return None
    current = [event for event in events if event.get("episode_id") == episode_id]
    window_lo, window_hi = anchor - 50_000_000, anchor + 400_000_000
    lanes: dict[int, list[int]] = defaultdict(list)
    image_tokens: list[int] = []
    for event in current:
        if event.get("kind") != "sse_chunk" or not (event.get("token_ids") or []):
            continue
        lane = event.get("lane")
        if lane is None:
            image_tokens.append(int(event["time_ns"]))
        else:
            lanes[int(lane)].extend([int(event["time_ns"])] * len(event["token_ids"]))
    gaps = []
    for times in lanes.values():
        times.sort()
        for before, after in zip(times, times[1:]):
            left, right = max(before, window_lo), min(after, window_hi)
            if right > left:
                gaps.append((right - left) / 1e6)
    max_gap = max(gaps) if gaps else None
    ttft = (min(image_tokens) - anchor) / 1e6 if image_tokens else None
    return {"episode_id": episode_id, "max_text_gap_ms": max_gap, "image_ttft_ms": ttft}


def select_f() -> None:
    by_profile: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    expected_groups = list(itertools.product(SIZES, STREAMS))
    for interval, budget, size, streams in itertools.product(INTERVALS, LM_BUDGETS, SIZES, STREAMS):
        prefix = f"Ccal-I{interval}-Q{budget}-L{size}-D{streams}"
        profile_rows: dict[int, dict[str, Any]] = {}
        for path in sorted((SSD / "results").glob(prefix + "-a*.json")):
            try:
                result = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            events = read_jsonl(SSD / "logs" / (path.stem + "-client-events.jsonl"))
            for row in result.get("episodes", []):
                trial = int(row.get("trial_seed", row.get("trial", -1)))
                if trial not in TRIAL_IDS or trial in profile_rows or not row_is_valid(row, streams):
                    continue
                metric = episode_metrics(row, events)
                if metric is None:
                    continue
                metric.update({"load": {"image_size": size, "text_streams": streams}, "trial_seed": trial})
                profile_rows[trial] = metric
        if len(profile_rows) != len(TRIAL_IDS):
            raise RuntimeError(f"calibration grid incomplete for {prefix}: have {sorted(profile_rows)}")
        by_profile[(interval, budget)].extend(profile_rows.values())

    summaries = []
    for (interval, budget), rows in sorted(by_profile.items()):
        joint = [
            row for row in rows
            if row["max_text_gap_ms"] is not None and row["image_ttft_ms"] is not None
            and row["max_text_gap_ms"] <= 30 and row["image_ttft_ms"] <= 55
        ]
        image_pass = [row for row in rows if row["image_ttft_ms"] is not None and row["image_ttft_ms"] <= 55]
        text_values = [float(row["max_text_gap_ms"]) for row in rows if row["max_text_gap_ms"] is not None]
        summaries.append({
            "prefill_interval": interval,
            "lm_budget": budget,
            "episodes": len(rows),
            "joint_passes": len(joint),
            "joint_rate": len(joint) / len(rows),
            "text_gap_p95_ms": percentile(text_values),
            "image_ttft_passes": len(image_pass),
            "image_ttft_pass_rate": len(image_pass) / len(rows),
            "episode_metrics": rows,
        })
    summaries.sort(key=lambda x: (
        -x["joint_rate"],
        x["text_gap_p95_ms"] if x["text_gap_p95_ms"] is not None else float("inf"),
        -x["image_ttft_pass_rate"],
        abs(x["prefill_interval"] - 1) + abs(x["lm_budget"] - 2048) / 1536,
    ))
    selected = summaries[0]
    report = {
        "method": "One global fixed native configuration selected on independent calibration episodes. Sort by pooled 30ms text-gap/55ms TTFT joint pass rate, then text-gap P95, image TTFT pass rate, then closeness to default interval=1/budget=2048.",
        "grid": {"prefill_interval": list(INTERVALS), "lm_budget": list(LM_BUDGETS), "load_groups": [list(x) for x in expected_groups], "episodes_per_profile": 12},
        "selected_F": {"prefill_interval": selected["prefill_interval"], "lm_budget": selected["lm_budget"]},
        "profiles": summaries,
        "calibration_episode_total": sum(x["episodes"] for x in summaries),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    OUT.write_text(serialized)
    print(json.dumps({
        "selected_F": report["selected_F"],
        "calibration_episode_total": report["calibration_episode_total"],
        "profiles": [{k: p[k] for k in ("prefill_interval", "lm_budget", "episodes", "joint_passes", "joint_rate", "text_gap_p95_ms", "image_ttft_pass_rate")} for p in summaries],
        "out": str(OUT),
    }, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    groups = list(itertools.product(INTERVALS, LM_BUDGETS, SIZES, STREAMS))
    compatibility_edge = (1, 512, 768, 6)
    groups.remove(compatibility_edge)
    random.Random(20261107).shuffle(groups)
    groups.insert(0, compatibility_edge)
    total = len(groups)
    for index, (interval, budget, size, streams) in enumerate(groups, 1):
        prefix = f"Ccal-I{interval}-Q{budget}-L{size}-D{streams}"
        if valid_trials(prefix, streams) == set(TRIAL_IDS):
            append_progress({"kind": "group_already_complete", "prefix": prefix, "index": index, "total": total})
            continue
        print(f"calibration group {index}/{total}: I={interval} Q={budget} image={size} text={streams}", flush=True)
        run_group(interval, budget, size, streams)
        print(f"calibration group complete {index}/{total}: {prefix}", flush=True)
    select_f()


if __name__ == "__main__":
    main()
