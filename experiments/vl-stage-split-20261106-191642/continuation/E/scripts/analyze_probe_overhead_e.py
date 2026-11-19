#!/usr/bin/env python3
"""Summarize paired client timings for E probe-off/queue-only/full-probe runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


PROFILES = ("probe-off", "queue-only", "full-probe")
METRICS = ("first_content_ms", "full_response_ms", "http_headers_ms")


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{index}: invalid JSON: {exc}") from exc
    return rows


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": ordered[round((len(ordered) - 1) * 0.95)],
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--event-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    samples: dict[str, dict[tuple[str, int], dict[str, Any]]] = defaultdict(dict)
    file_meta = []
    for path in args.inputs:
        rows = read_rows(path)
        meta = next((row for row in rows if row.get("kind") == "calibration_metadata"), {})
        profile = str(meta.get("profile", ""))
        block_id = str(meta.get("block_id", ""))
        if profile not in PROFILES or not block_id:
            raise ValueError(f"missing profile/block metadata in {path}")
        file_meta.append({"path": str(path), "profile": profile, "block_id": block_id})
        for row in rows:
            if row.get("kind") != "request_timing" or row.get("warmup"):
                continue
            index = int(row["request_index"])
            key = (block_id, index)
            if key in samples[profile]:
                raise ValueError(f"duplicate paired sample {profile} {key}")
            samples[profile][key] = row

    common_keys = set.intersection(*(set(samples[profile]) for profile in PROFILES))
    content_mismatches = []
    pairs = []
    pair_deltas: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for key in sorted(common_keys):
        rows = {profile: samples[profile][key] for profile in PROFILES}
        signatures = {
            profile: hashlib.sha256(str(rows[profile].get("content", "")).encode()).hexdigest()
            for profile in PROFILES
        }
        if len(set(signatures.values())) != 1:
            content_mismatches.append({"pair": list(key), "text_hashes": signatures})
        pair = {"block_id": key[0], "request_index": key[1], "profiles": {}}
        for profile in PROFILES:
            pair["profiles"][profile] = {
                "status": rows[profile].get("status"),
                "first_content_ms": rows[profile].get("first_content_ms"),
                "full_response_ms": rows[profile].get("full_response_ms"),
                "http_headers_ms": rows[profile].get("http_headers_ms"),
                "usage": rows[profile].get("usage"),
                "content_sha256": signatures[profile],
            }
        for treatment, reference in (("queue-only", "probe-off"), ("full-probe", "probe-off"), ("full-probe", "queue-only")):
            comparison = f"{treatment}_minus_{reference}"
            for metric in METRICS:
                a = rows[treatment].get(metric)
                b = rows[reference].get(metric)
                if isinstance(a, (float, int)) and isinstance(b, (float, int)):
                    pair_deltas[comparison][metric].append(float(a) - float(b))
        pairs.append(pair)

    event_counts = {}
    for profile in PROFILES:
        paths = sorted(args.event_dir.glob(f"*-{profile}-server-events.jsonl"))
        rows = []
        for path in paths:
            rows.extend(read_rows(path))
        from collections import Counter
        event_counts[profile] = {
            "files": [str(path) for path in paths],
            "rows": len(rows),
            "kinds": dict(Counter(row.get("kind", "unknown") for row in rows)),
            "queue_phases": dict(Counter(row.get("phase", "unknown") for row in rows if row.get("kind") == "queue_timing")),
        }

    profile_stats = {}
    for profile in PROFILES:
        profile_stats[profile] = {
            metric: stats([
                float(row[metric])
                for row in samples[profile].values()
                if isinstance(row.get(metric), (int, float))
            ])
            for metric in METRICS
        }
        profile_stats[profile]["http_200_count"] = sum(row.get("status") == 200 for row in samples[profile].values())
        profile_stats[profile]["measured_sample_count"] = len(samples[profile])

    result = {
        "schema_version": 1,
        "files": file_meta,
        "measured_pairs": len(pairs),
        "profile_stats": profile_stats,
        "paired_delta_stats_ms": {
            comparison: {metric: stats(values) for metric, values in metrics.items()}
            for comparison, metrics in pair_deltas.items()
        },
        "event_counts": event_counts,
        "content_mismatches": content_mismatches,
        "pairs": pairs,
        "interpretation": "This estimates the additional timing/trace path overhead of the E patched overlay under this one synthetic workload. probe-off still imports the E overlay and has hook-call/gate costs; it is not a stock unpatched-vLLM baseline. Client first-content and full-response times include model/runtime variation and are not formal latency results.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "analyzed",
        "measured_pairs": result["measured_pairs"],
        "profile_stats": profile_stats,
        "paired_delta_stats_ms": result["paired_delta_stats_ms"],
        "event_counts": event_counts,
        "content_mismatch_count": len(content_mismatches),
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
