#!/usr/bin/env python3
"""Build an auditable E2 summary from complete and partial per-run artifacts."""
from __future__ import annotations

import json
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

STAGE = Path(__file__).resolve().parent
OUT = STAGE / "e2-paper-normal-v1"
WORKLOADS = STAGE / "workloads"
METRICS = [
    "raw_output_tok_s",
    "effective_tok_s",
    "ttft_mean_s",
    "ttft_p99_s",
    "e2e_mean_s",
    "stall_fraction",
]


def read_json(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    driver_summary = OUT / "summary.driver-raw.json"
    summary_path = OUT / "summary.json"
    if summary_path.exists() and not driver_summary.exists():
        shutil.copy2(summary_path, driver_summary)

    progress = read_json(OUT / "progress.json")
    failures = read_json(OUT / "failures.json") if (OUT / "failures.json").exists() else []
    by_profile = defaultdict(list)
    by_seed_profile = {}
    for row in progress:
        by_profile[row["profile"]].append(row)
        by_seed_profile[(row["seed"], row["profile"])] = row

    failure_rows = []
    failure_request_details = []
    for failure in failures:
        run = failure.get("run", {})
        profile = failure["profile"]
        seed = failure["seed"]
        failure_rows.append({
            "tag": failure["tag"],
            "seed": seed,
            "profile": profile,
            "requested": run.get("requests", 0),
            "completed": run.get("passed", 0),
            "wall_s": run.get("wall_s"),
            "error": failure.get("error", failure.get("stage")),
        })
        workload_path = WORKLOADS / f"e2-paper-normal-adaptation-s{seed}.json"
        workload = read_json(workload_path)["requests"]
        raw_path = OUT / "raw" / failure["tag"] / "requests.jsonl"
        with raw_path.open() as stream:
            rows = [json.loads(line) for line in stream]
        for row in rows:
            if row.get("ok"):
                continue
            request = workload[row["id"]]
            events = row.get("events", [])
            last = events[-1] if events else {}
            failure_request_details.append({
                "tag": failure["tag"],
                "seed": seed,
                "profile": profile,
                "request_id": row["id"],
                "input_tokens": len(request["input_ids"]),
                "requested_output_tokens": request["max_new_tokens"],
                "last_reported_completion_tokens": last.get("data", {}).get("meta_info", {}).get("completion_tokens", 0),
                "last_event_s": last.get("time_s"),
                "error": row.get("error"),
            })

    per_profile = {}
    for profile in ("plain", "chunked", "tokenflow"):
        complete = by_profile[profile]
        partial = [r for r in failure_rows if r["profile"] == profile]
        profile_summary = {
            "complete_runs": len(complete),
            "incomplete_runs": len(partial),
            "requested_requests": 80 * (len(complete) + len(partial)),
            "completed_responses": sum(r["completed"] for r in complete) + sum(r["completed"] for r in partial),
            "fully_analyzed_requests": sum(r["requested"] for r in complete),
            "partial_run_completed_responses": sum(r["completed"] for r in partial),
            "incomplete_run_details": partial,
        }
        if complete:
            profile_summary["arithmetic_mean_of_complete_run_metrics"] = {
                metric: statistics.mean(row[metric] for row in complete)
                for metric in METRICS
            }
            profile_summary["per_seed"] = [
                {"seed": row["seed"], **{metric: row[metric] for metric in METRICS}}
                for row in sorted(complete, key=lambda row: row["seed"])
            ]
        per_profile[profile] = profile_summary

    paired_changes = {}
    for metric in METRICS:
        pairs = []
        for seed in (101, 102, 103):
            plain = by_seed_profile.get((seed, "plain"))
            tokenflow = by_seed_profile.get((seed, "tokenflow"))
            if plain and tokenflow:
                pairs.append({
                    "seed": seed,
                    "plain": plain[metric],
                    "tokenflow": tokenflow[metric],
                    "change_pct_tokenflow_vs_plain": (tokenflow[metric] / plain[metric] - 1) * 100,
                })
        paired_changes[metric] = {
            "mean_change_pct_tokenflow_vs_plain": statistics.mean(p["change_pct_tokenflow_vs_plain"] for p in pairs),
            "per_seed": pairs,
        }

    requested = sum(summary["requested_requests"] for summary in per_profile.values())
    completed = sum(summary["completed_responses"] for summary in per_profile.values())
    result = {
        "case": "E2",
        "suite": "paper-normal-adaptation",
        "attempted_runs": len(progress) + len(failure_rows),
        "complete_runs": len(progress),
        "incomplete_runs": len(failure_rows),
        "requested_requests": requested,
        "completed_responses": completed,
        "fully_analyzed_requests": sum(row["requested"] for row in progress),
        "partial_run_completed_responses": sum(row["completed"] for row in failure_rows),
        "per_profile": per_profile,
        "paired_tokenflow_vs_plain_change": paired_changes,
        "failure_requests": failure_request_details,
        "interpretation": "Adaptation experiment only. The paper's exact experiment commit, normal-distribution parameters, chunked baseline CLI, and per-figure prompt corpus are not fully disclosed. Means and paired changes use complete 80/80 runs only; partial runs are reported separately.",
    }
    summary_path.write_text(json.dumps(result, indent=2))
    (OUT / "incomplete-request-details.json").write_text(json.dumps(failure_request_details, indent=2))
    print(json.dumps({
        "attempted_runs": result["attempted_runs"],
        "complete_runs": result["complete_runs"],
        "incomplete_runs": result["incomplete_runs"],
        "requested_requests": requested,
        "completed_responses": completed,
        "profile_counts": {k: {"complete_runs": v["complete_runs"], "incomplete_runs": v["incomplete_runs"], "completed_responses": v["completed_responses"]} for k, v in per_profile.items()},
        "paired_tokenflow_vs_plain_change": {k: v["mean_change_pct_tokenflow_vs_plain"] for k, v in paired_changes.items()},
        "failure_requests": failure_request_details,
    }, indent=2))


if __name__ == "__main__":
    main()
