#!/usr/bin/env python3
"""Summarize request-level pilot SSE logs using the frozen SLO thresholds."""

import argparse
import json
import math
from pathlib import Path


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def read_requests(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def summarize(path: Path, short_ttft: float, long_ttft: float, interval_ms: float) -> dict:
    requests = read_requests(path)
    all_intervals = []
    send_lateness = []
    per_request = []
    ttft_by_class = {"short": [], "long": []}
    satisfied = 0
    complete = 0
    batched_events = 0
    count_increments = 0
    long_ttft_with_concurrent_output = 0

    for request in requests:
        lateness = request.get("send_lateness_s")
        if lateness is not None:
            send_lateness.append(lateness * 1000)
        if request["class"] != "long" or request.get("first_token_s") is None:
            continue
        prefill_interval_start = request["send_s"]
        prefill_interval_end = prefill_interval_start + request["first_token_s"]
        overlapping_output = any(
            other["id"] != request["id"]
            and other.get("send_s", float("inf")) <= prefill_interval_end
            and other.get("end_s", -1) >= prefill_interval_start
            and any(
                prefill_interval_start <= event.get("time_s", -1) <= prefill_interval_end
                and event.get("data", {}).get("meta_info", {}).get("completion_tokens", 0) > 0
                for event in other.get("events", [])
            )
            for other in requests
        )
        long_ttft_with_concurrent_output += int(overlapping_output)

    boundaries = sorted(
        [
            (request.get("send_s", 0), 1)
            for request in requests
        ]
        + [
            (request.get("end_s", 0), -1)
            for request in requests
        ]
    )
    active = 0
    max_client_inflight = 0
    for _, delta in boundaries:
        active += delta
        max_client_inflight = max(max_client_inflight, active)

    for request in requests:
        previous_count = 0
        previous_time = None
        intervals = []
        for event in request.get("events", []):
            data = event.get("data", {})
            count = data.get("meta_info", {}).get("completion_tokens", 0)
            if count > previous_count:
                count_increments += 1
                if count - previous_count > 1:
                    batched_events += 1
                if previous_time is not None:
                    gap = (event["time_s"] - previous_time) * 1000
                    intervals.append(gap)
                    all_intervals.append(gap)
                previous_time = event["time_s"]
                previous_count = count

        request_p95 = percentile(intervals, 0.95)
        if request.get("done") and request.get("completion_tokens", 0) == 256:
            complete += 1
        first_token = request.get("first_token_s")
        if first_token is not None:
            ttft_by_class[request["class"]].append(first_token)
        threshold = short_ttft if request["class"] == "short" else long_ttft
        passed = (
            request.get("ok")
            and first_token is not None
            and first_token <= threshold
            and request_p95 is not None
            and request_p95 <= interval_ms
        )
        satisfied += int(bool(passed))
        per_request.append(
            {
                "id": request["id"],
                "class": request["class"],
                "ok": request.get("ok"),
                "ttft_s": first_token,
                "output_interval_p95_ms": request_p95,
                "slo_pass": bool(passed),
            }
        )

    window = max((item.get("end_s", 0) for item in requests), default=0)
    return {
        "requests": len(requests),
        "complete": complete,
        "successful_slo_requests": satisfied,
        "arrival_to_last_terminal_window_s": window,
        "completion_throughput_rps": complete / window if window else 0,
        "goodput_rps": satisfied / window if window else 0,
        "send_lateness_ms": {
            "p50": percentile(send_lateness, 0.50),
            "p95": percentile(send_lateness, 0.95),
            "max": max(send_lateness, default=None),
        },
        "max_client_inflight": max_client_inflight,
        "long_ttft_with_concurrent_output": {
            "n": long_ttft_with_concurrent_output,
            "total": sum(item["class"] == "long" for item in requests),
        },
        "ttft_s": {
            kind: {
                "n": len(values),
                "p50": percentile(values, 0.50),
                "p95": percentile(values, 0.95),
            }
            for kind, values in ttft_by_class.items()
        },
        "output_event_interval_ms": {
            "n": len(all_intervals),
            "p95": percentile(all_intervals, 0.95),
            "p99": percentile(all_intervals, 0.99),
            "events_with_multiple_tokens_fraction": (
                batched_events / count_increments if count_increments else None
            ),
        },
        "per_request": per_request,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("requests_jsonl")
    parser.add_argument("--short-ttft", type=float, required=True)
    parser.add_argument("--long-ttft", type=float, required=True)
    parser.add_argument("--interval-ms", type=float, required=True)
    args = parser.parse_args()
    result = summarize(
        Path(args.requests_jsonl),
        args.short_ttft,
        args.long_ttft,
        args.interval_ms,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
