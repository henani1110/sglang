#!/usr/bin/env python3
"""Join queue timing phases to first scheduler admission and summarize durations."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None}
    ordered = sorted(values)

    def percentile(p: float) -> float:
        index = round((len(ordered) - 1) * p)
        return ordered[index]

    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-events", type=Path, required=True)
    parser.add_argument("--scheduler-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    queue_rows = [row for row in read_jsonl(args.server_events) if row.get("kind") == "queue_timing"]
    phases: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in queue_rows:
        request_id = str(row.get("request_id", ""))
        phase = str(row.get("phase", ""))
        if request_id and phase:
            phases[request_id].setdefault(phase, row)

    first_schedule: dict[str, int] = {}
    for row in read_jsonl(args.scheduler_events):
        timestamp = row.get("time_ns")
        if not isinstance(timestamp, int):
            continue
        for req in row.get("scheduled_new_reqs", []) or []:
            if isinstance(req, dict) and req.get("request_id") is not None:
                first_schedule.setdefault(str(req["request_id"]), timestamp)

    records = []
    by_phase: dict[str, list[float]] = defaultdict(list)
    complete_queue_ms: list[float] = []
    misses = []
    for request_id, request_phases in sorted(phases.items()):
        ingress = request_phases.get("api_ingress", {}).get("time_ns")
        render = request_phases.get("api_render_complete", {}).get("time_ns")
        input_start = request_phases.get("engine_input_start", {}).get("time_ns")
        input_ready = request_phases.get("engine_input_ready", {}).get("time_ns")
        submit_row = request_phases.get("engine_submit", {})
        submit = submit_row.get("time_ns")
        engine_request_id = str(submit_row.get("engine_request_id", ""))
        scheduled = first_schedule.get(engine_request_id)
        deltas: dict[str, float | None] = {
            "api_render_ms": None,
            "engine_input_processing_ms": None,
            "local_handoff_ms": None,
            "submit_to_first_schedule_ms": None,
            "ingress_to_first_schedule_ms": None,
        }
        pairs = [
            ("api_render_ms", ingress, render),
            ("engine_input_processing_ms", input_start, input_ready),
            ("local_handoff_ms", input_ready, submit),
            ("submit_to_first_schedule_ms", submit, scheduled),
            ("ingress_to_first_schedule_ms", ingress, scheduled),
        ]
        for name, start, end in pairs:
            if isinstance(start, int) and isinstance(end, int):
                value = (end - start) / 1_000_000
                deltas[name] = value
                by_phase[name].append(value)
                if value < 0:
                    misses.append({"request_id": request_id, "metric": name, "value_ms": value})
        if isinstance(deltas["submit_to_first_schedule_ms"], float):
            complete_queue_ms.append(deltas["submit_to_first_schedule_ms"])
        records.append({
            "request_id": request_id,
            "engine_request_id": engine_request_id or None,
            "phases_present": sorted(request_phases),
            "first_schedule_seen": isinstance(scheduled, int),
            **deltas,
        })

    result = {
        "schema_version": 1,
        "server_event_file": str(args.server_events),
        "scheduler_event_file": str(args.scheduler_events),
        "queue_event_rows": len(queue_rows),
        "request_count": len(records),
        "scheduled_request_count": sum(bool(record["first_schedule_seen"]) for record in records),
        "summaries": {name: summarize(values) for name, values in by_phase.items()},
        "negative_duration_rows": misses,
        "records": records,
        "interpretation": "submit_to_first_schedule includes EngineCore IPC and scheduler-cycle wait; scheduler time is recorded after schedule() returns. perf_counter_ns is host monotonic and is comparable across local processes on this host.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "joined",
        "queue_event_rows": result["queue_event_rows"],
        "request_count": result["request_count"],
        "scheduled_request_count": result["scheduled_request_count"],
        "summaries": result["summaries"],
        "negative_duration_rows": len(misses),
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
