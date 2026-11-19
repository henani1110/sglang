#!/usr/bin/env python3
"""Analyze the frozen D formal matrix from client, scheduler and server traces."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Iterable


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL {path}:{line_no}: {exc}") from exc


def quantile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(x) for x in values if math.isfinite(float(x)))
    if not ordered:
        return None
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def safe_rate(n: int, d: int) -> float | None:
    return n / d if d else None


def resolve_request_id(key: str, known: set[str]) -> str | None:
    if key in known:
        return key
    # vLLM scheduler/decision events may append an internal hexadecimal
    # suffix to the public API request ID (for example, ``chatcmpl-...-a1b2c3d4``).
    # Strip one trailing component at a time and only accept an exact match in
    # the known request set. This also handles older numeric suffixes without
    # accidentally mapping an unknown request to a merely similar ID.
    parts = key.split("-")
    while len(parts) > 1:
        parts.pop()
        candidate = "-".join(parts)
        if candidate in known:
            return candidate
    return None


def text_gap_data(
    client_rows: list[dict[str, Any]], start_ns: int, end_ns: int
) -> tuple[list[float], dict[int, list[tuple[int, int, float]]], dict[int, int]]:
    token_times: dict[int, list[int]] = collections.defaultdict(list)
    token_counts: dict[int, int] = collections.Counter()
    for row in client_rows:
        if row.get("kind") != "text_response" or row.get("lane") is None:
            continue
        lane = int(row["lane"])
        times = row.get("token_times_ns") or []
        ids = row.get("token_ids") or []
        for stamp in times:
            token_times[lane].append(int(stamp))
            if start_ns <= int(stamp) < end_ns:
                token_counts[lane] += 1

    all_gaps: list[float] = []
    intervals: dict[int, list[tuple[int, int, float]]] = {}
    for lane, stamps in token_times.items():
        ordered = sorted(stamps)
        lane_rows: list[tuple[int, int, float]] = []
        for before, after in zip(ordered, ordered[1:]):
            if after < start_ns or before > end_ns or after < before:
                continue
            gap = (after - before) / 1e6
            lane_rows.append((before, after, gap))
            all_gaps.append(gap)
        intervals[lane] = lane_rows
    return all_gaps, intervals, token_counts


def overlap_text_max(
    intervals: dict[int, list[tuple[int, int, float]]],
    lane_ids: set[int],
    start_ns: int,
    end_ns: int,
) -> tuple[bool, float | None]:
    maxima: list[float] = []
    for lane in lane_ids:
        lane_gaps = [
            gap for before, after, gap in intervals.get(lane, [])
            if before <= end_ns and after >= start_ns
        ]
        if not lane_gaps:
            return False, None
        maxima.append(max(lane_gaps))
    return bool(maxima) and all(x <= math.inf for x in maxima), max(maxima) if maxima else None


def collect_scheduler(
    path: Path,
    image_ids: set[str],
    text_ids: set[str],
    start_ns: int,
    end_ns: int,
) -> tuple[dict[str, int], dict[str, int], dict[str, list[int]]]:
    encoder_step: dict[str, int] = {}
    prefill_step: dict[str, int] = {}
    text_decode: dict[str, list[int]] = collections.defaultdict(list)
    all_ids = image_ids | text_ids
    for row in iter_jsonl(path):
        if row.get("kind") != "scheduler_step":
            continue
        stamp = int(row["time_ns"])
        enc = row.get("scheduled_encoder_inputs", {}) or {}
        for key in enc:
            rid = resolve_request_id(str(key), image_ids)
            if rid is not None:
                encoder_step.setdefault(rid, stamp)
        scheduled = row.get("num_scheduled_tokens", {}) or {}
        for key, count in scheduled.items():
            rid = resolve_request_id(str(key), all_ids)
            if rid is None:
                continue
            count = int(count)
            if rid in image_ids and count > 0:
                prefill_step.setdefault(rid, stamp)
            elif rid in text_ids and count == 1 and start_ns <= stamp <= end_ns:
                text_decode[rid].append(stamp)
    return encoder_step, prefill_step, text_decode


def collect_server(path: Path, image_ids: set[str], start_ns: int, end_ns: int) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[float], dict[str, Any]]:
    engine: dict[str, dict[str, Any]] = {}
    api: dict[str, dict[str, Any]] = {}
    visual_cpu: list[float] = []
    counts = collections.Counter()
    for row in iter_jsonl(path):
        kind = row.get("kind")
        counts[str(kind)] += 1
        if kind == "visual_forward":
            try:
                value = float(row.get("cpu_elapsed_ms", 0.0))
                stamp = int(row.get("time_ns", 0))
                if start_ns <= stamp < end_ns and value > 0 and math.isfinite(value):
                    visual_cpu.append(value)
            except (TypeError, ValueError):
                pass
        rid = str(row.get("request_id", ""))
        if rid not in image_ids:
            continue
        if kind == "engine_result_ready_to_api" and int(row.get("token_index_start", -1)) == 0:
            if rid not in engine or int(row["time_ns"]) < int(engine[rid]["time_ns"]):
                engine[rid] = row
        elif kind == "api_yield" and int(row.get("token_index_start", -1)) == 0:
            if rid not in api or int(row["time_ns"]) < int(api[rid]["time_ns"]):
                api[rid] = row
    return engine, api, visual_cpu, dict(counts)


def decode_gaps(
    decode_steps: dict[str, list[int]],
    id_to_lane: dict[str, int],
) -> tuple[list[float], dict[int, list[tuple[int, int, float]]]]:
    by_lane: dict[int, list[int]] = collections.defaultdict(list)
    for request_id, stamps in decode_steps.items():
        lane = id_to_lane.get(request_id)
        if lane is None:
            continue
        ordered = sorted(set(stamps))
        by_lane[lane].extend(ordered)
    all_gaps: list[float] = []
    intervals: dict[int, list[tuple[int, int, float]]] = {}
    for lane, stamps in by_lane.items():
        ordered = sorted(set(stamps))
        lane_rows = []
        for before, after in zip(ordered, ordered[1:]):
            if after <= before:
                continue
            gap = (after - before) / 1e6
            lane_rows.append((before, after, gap))
            all_gaps.append(gap)
        intervals[lane] = lane_rows
    return all_gaps, intervals


def run_analysis(
    row: dict[str, Any],
    profile: dict[str, Any],
    thresholds: tuple[tuple[float, float], ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = read_json(Path(row["result_path"]))
    if result.get("status") != "completed":
        raise RuntimeError(f"completed manifest row has non-completed result: {row.get('run_id')}")
    start_ns = int(result["measurement_start_ns"])
    measure_s = int(result["measurement_seconds"])
    end_ns = start_ns + measure_s * 1_000_000_000
    client_path = Path(row["client_events_path"])
    client_rows = list(iter_jsonl(client_path))
    image_sends = [
        x for x in client_rows
        if x.get("kind") == "image_request_send" and x.get("phase") == "measurement"
    ]
    image_responses = [
        x for x in client_rows
        if x.get("kind") == "image_response" and x.get("phase") == "measurement"
    ]
    text_responses = [x for x in client_rows if x.get("kind") == "text_response"]
    text_ids = {
        str(x["request_id"]) for x in text_responses
        if x.get("request_id") and any(start_ns <= int(t) < end_ns for t in (x.get("token_times_ns") or []))
    }
    id_to_lane = {
        str(x["request_id"]): int(x["lane"]) for x in text_responses
        if x.get("request_id") and x.get("lane") is not None
    }
    measured_text_ids = text_ids & set(id_to_lane)
    image_ids = {str(x["request_id"]) for x in image_responses if x.get("request_id")}
    scheduler_path = Path(row["scheduler_events_path"])
    server_path = Path(row["server_events_path"])
    e_steps, p_steps, decode_steps = collect_scheduler(
        scheduler_path, image_ids, measured_text_ids, start_ns, end_ns
    )
    decode_gap_values, decode_gap_by_lane = decode_gaps(decode_steps, id_to_lane)
    client_gaps, text_gap_by_lane, token_counts = text_gap_data(client_rows, start_ns, end_ns)
    engine_rows, api_rows, visual_cpu, server_counts = collect_server(server_path, image_ids, start_ns, end_ns)

    grid = int(row["image_size"])
    lanes = int(row["text_streams"])
    cell = profile["cells"][f"{grid}:{lanes}"]
    e_hat = float(cell["e_hat_ms"])
    p_hat = float(cell["p_hat_ms"])
    d_hat = float(cell["decode_step_hat_ms"])
    split_hat = float(cell["split_gap_hat_ms"])
    session = (
        f"Dformal-{row['policy']}-G{grid}-L{lanes}-{row['arrival_mode']}"
        f"-B{int(row.get('block', 1))}-{row.get('attempt', 'a0')}"
    )
    decision_name = f"{session}-decisions.jsonl"
    # run_server_d.sh writes decisions under D/logs/, while server events live
    # under D/logs/formal/. Prefer that canonical location, with a fallback for
    # older runs that colocated the files.
    decision_path = server_path.parent.parent / decision_name
    if not decision_path.exists():
        decision_path = server_path.parent / decision_name
    decision_rows = list(iter_jsonl(decision_path)) if row["policy"] == "A" and decision_path.exists() else []
    decision_rows = [
        x for x in decision_rows
        if str(x.get("request_id")) in image_ids
        or start_ns <= int(x.get("time_ns", 0)) < end_ns
    ]
    decision_by_id = {str(x.get("request_id")): x for x in decision_rows}

    response_by_label = {str(x.get("label")): x for x in image_responses}
    sends_by_label = {str(x.get("request_label")): x for x in image_sends}
    image_metrics: list[dict[str, Any]] = []
    queue_ms: list[float] = []
    e_to_p_ms: list[float] = []
    engine_after_p_ms: list[float] = []
    engine_to_api_ms: list[float] = []
    api_to_client_ms: list[float] = []
    combined_prediction_errors: list[float] = []
    p_prediction_errors: list[float] = []
    split_observed: list[float] = []
    split_prediction_errors: list[float] = []
    send_jitter_ms: list[float] = []
    completion = timeout = incomplete = image_error_count = 0
    lane_ids = set(range(lanes))
    for label, send in sends_by_label.items():
        response = response_by_label.get(label)
        send_ns = int(send.get("actual_send_ns", 0))
        plan_ns = int(send.get("planned_ns", send_ns))
        send_jitter_ms.append((send_ns - plan_ns) / 1e6)
        request_id = str(response.get("request_id")) if response and response.get("request_id") else ""
        first_ns = int(response["first_token_ns"]) if response and response.get("first_token_ns") else None
        ttft = (first_ns - send_ns) / 1e6 if first_ns is not None else None
        error = response.get("error") if response else "missing_response"
        token_count = len(response.get("token_ids") or []) if response else 0
        is_complete = response is not None and not error and response.get("finish_reason") == "length" and token_count >= 16
        if is_complete:
            completion += 1
        elif error:
            image_error_count += 1
            if "timeout" in str(error).lower():
                timeout += 1
            else:
                incomplete += 1
        else:
            incomplete += 1

        e_ns = e_steps.get(request_id)
        p_ns = p_steps.get(request_id)
        eng = engine_rows.get(request_id)
        api = api_rows.get(request_id)
        decision = decision_by_id.get(request_id, {})
        local_e_hat = float(decision.get("e_hat_ms", e_hat))
        local_p_hat = float(decision.get("p_hat_ms", p_hat))
        local_split_hat = float(decision.get("split_gap_hat_ms", split_hat))
        eng_ns = int(eng["time_ns"]) if eng else None
        api_ns = int(api["time_ns"]) if api else None
        client_ids = [int(x) for x in (response.get("token_ids") or [])] if response else []
        engine_ids = [int(x) for x in (eng.get("token_ids") or [])] if eng else []
        api_ids = [int(x) for x in (api.get("token_ids") or [])] if api else []
        token_id_match = bool(
            client_ids and engine_ids and api_ids
            and client_ids[0] == engine_ids[0] == api_ids[0]
        )
        q = (e_ns - send_ns) / 1e6 if e_ns is not None else None
        ep = (p_ns - e_ns) / 1e6 if e_ns is not None and p_ns is not None else None
        after_p = (eng_ns - p_ns) / 1e6 if eng_ns is not None and p_ns is not None else None
        if q is not None:
            queue_ms.append(q)
        if ep is not None:
            e_to_p_ms.append(ep)
            observed_gap = max(0.0, ep - local_e_hat)
            selected_action = decision_by_id.get(request_id, {}).get("selected_action", "S" if row["policy"] == "S" else "N")
            if ep > 0 and selected_action == "S":
                split_observed.append(observed_gap)
                split_prediction_errors.append(observed_gap - local_split_hat)
        if after_p is not None:
            engine_after_p_ms.append(after_p)
            selected_action = decision_by_id.get(request_id, {}).get("selected_action", "S" if row["policy"] == "S" else "N")
            if selected_action == "S" and p_ns is not None and e_ns is not None and p_ns > e_ns:
                p_prediction_errors.append(after_p - local_p_hat)
            else:
                combined_prediction_errors.append(after_p - (local_e_hat + local_p_hat))
        if eng_ns is not None and api_ns is not None:
            engine_to_api_ms.append((api_ns - eng_ns) / 1e6)
        if api_ns is not None and first_ns is not None:
            api_to_client_ms.append((first_ns - api_ns) / 1e6)

        joint_max_by_lane: dict[int, float] = {}
        if first_ns is not None:
            for lane in lane_ids:
                overlaps = [
                    gap for before, after, gap in text_gap_by_lane.get(lane, [])
                    if before <= first_ns and after >= send_ns
                ]
                if overlaps:
                    joint_max_by_lane[lane] = max(overlaps)
        event_order = bool(
            eng_ns is not None and api_ns is not None and first_ns is not None
            and eng_ns <= api_ns <= first_ns and token_id_match
        )
        image_metrics.append(
            {
                "label": label,
                "request_id": request_id or None,
                "image_id": send.get("image_id"),
                "image_sha256": send.get("image_sha256"),
                "planned_ns": plan_ns,
                "actual_send_ns": send_ns,
                "send_jitter_ms": (send_ns - plan_ns) / 1e6,
                "first_token_ns": first_ns,
                "ttft_ms": ttft,
                "error": error,
                "finish_reason": response.get("finish_reason") if response else None,
                "output_tokens": token_count,
                "completed": is_complete,
                "encoder_step_ns": e_ns,
                "prefill_step_ns": p_ns,
                "engine_ready_ns": eng_ns,
                "api_yield_ns": api_ns,
                "queue_wait_ms": q,
                "e_to_p_scheduler_gap_ms": ep,
                "engine_after_prefill_ms": after_p,
                "engine_to_api_ms": (api_ns - eng_ns) / 1e6 if eng_ns is not None and api_ns is not None else None,
                "api_to_client_ms": (first_ns - api_ns) / 1e6 if api_ns is not None and first_ns is not None else None,
                "trace_order_valid": event_order,
                "first_token_id_match": token_id_match,
                "overlap_text_max_gap_by_lane_ms": joint_max_by_lane,
            }
        )

    run_image_ttft = [float(x["ttft_ms"]) for x in image_metrics if x["ttft_ms"] is not None]
    run_queue = [float(x) for x in queue_ms if math.isfinite(x)]
    run_ep = [float(x) for x in e_to_p_ms if math.isfinite(x)]
    run_after_p = [float(x) for x in engine_after_p_ms if math.isfinite(x)]

    text_started = [
        x for x in text_responses
        if start_ns <= int(x.get("started_ns", 0)) < end_ns
    ]
    text_done = text_timeout = text_errors = text_incomplete = 0
    for x in text_started:
        error = x.get("error")
        ids = x.get("token_ids") or []
        complete = not error and x.get("finish_reason") == "length" and len(ids) >= 192
        if complete:
            text_done += 1
        elif error:
            text_errors += 1
            if "timeout" in str(error).lower():
                text_timeout += 1
            elif "cancel" in str(error).lower():
                text_incomplete += 1
            else:
                text_incomplete += 1
        else:
            text_incomplete += 1

    sensitivity: dict[str, Any] = {}
    for text_limit, image_limit in thresholds:
        passes = 0
        measured = 0
        image_passes = 0
        for image in image_metrics:
            measured += 1
            ttft = image.get("ttft_ms")
            image_ok = ttft is not None and float(ttft) <= image_limit
            if image_ok:
                image_passes += 1
            max_by_lane = image.get("overlap_text_max_gap_by_lane_ms", {})
            text_ok = len(max_by_lane) == lanes and all(
                float(v) <= text_limit for v in max_by_lane.values()
            )
            if (text_limit, image_limit) == (30.0, 55.0):
                image["joint_30_55_pass"] = bool(image_ok and text_ok)
            if image_ok and text_ok:
                passes += 1
        sensitivity[f"{text_limit:g}/{image_limit:g}"] = {
            "joint_image_requests_passed": passes,
            "image_requests_measured": measured,
            "joint_pass_rate": safe_rate(passes, measured),
            "joint_goodput_per_s": passes / measure_s,
            "image_ttft_passed": image_passes,
            "image_goodput_per_s": image_passes / measure_s,
        }

    metric = {
        "run_id": row.get("run_id"),
        "policy": row["policy"],
        "grid": grid,
        "text_streams": lanes,
        "arrival_mode": row["arrival_mode"],
        "seed": int(row["seed"]),
        "block": int(row.get("block", 0)),
        "attempt": row.get("attempt"),
        "status": row.get("status"),
        "measurement_seconds": measure_s,
        "planned_image_requests": len(image_sends),
        "image_response_rows": len(image_responses),
        "image_completed": completion,
        "image_errors": image_error_count,
        "image_timeouts": timeout,
        "image_incomplete": incomplete,
        "image_completion_rate": safe_rate(completion, len(image_sends)),
        "image_error_rate": safe_rate(image_error_count, len(image_sends)),
        "text_requests_started_in_measurement": len(text_started),
        "text_completed": text_done,
        "text_errors": text_errors,
        "text_timeouts": text_timeout,
        "text_incomplete": text_incomplete,
        "text_completion_rate": safe_rate(text_done, len(text_started)),
        "text_gap_event_count": len(client_gaps),
        "text_gap_p50_ms": quantile(client_gaps, 0.50),
        "text_gap_p95_ms": quantile(client_gaps, 0.95),
        "text_gap_p99_ms": quantile(client_gaps, 0.99),
        "text_max_gap_ms": max(client_gaps) if client_gaps else None,
        "text_gap_events_le_30ms": sum(x <= 30.0 for x in client_gaps),
        "text_intertoken_gaps_le_30_per_s": sum(x <= 30.0 for x in client_gaps) / measure_s,
        "image_ttft_p50_ms": quantile(run_image_ttft, 0.50),
        "image_ttft_p95_ms": quantile(run_image_ttft, 0.95),
        "image_ttft_p99_ms": quantile(run_image_ttft, 0.99),
        "image_queue_wait_p50_ms": quantile(run_queue, 0.50),
        "image_queue_wait_p95_ms": quantile(run_queue, 0.95),
        "e_to_p_scheduler_gap_p95_ms": quantile(run_ep, 0.95),
        "engine_after_prefill_p95_ms": quantile(run_after_p, 0.95),
        "engine_to_api_p95_ms": quantile(engine_to_api_ms, 0.95),
        "api_to_client_p95_ms": quantile(api_to_client_ms, 0.95),
        "open_loop_send_jitter_p95_ms": quantile(send_jitter_ms, 0.95),
        "joint_30_55_goodput_per_s": sensitivity["30/55"]["joint_goodput_per_s"],
        "joint_30_55_pass_rate": sensitivity["30/55"]["joint_pass_rate"],
        "scheduler_decode_gap_p75_ms": quantile(decode_gap_values, 0.75),
        "profile_e_hat_ms": e_hat,
        "profile_p_hat_ms": p_hat,
        "profile_decode_step_hat_ms": d_hat,
        "profile_split_gap_hat_ms": split_hat,
        "formal_visual_forward_cpu_elapsed_p75_ms": quantile(visual_cpu, 0.75),
        "visual_forward_cpu_sample_count": len(visual_cpu),
        "decode_step_sample_count": len(decode_gap_values),
        "e_to_p_sample_count": len(run_ep),
        "observed_split_gap_p75_ms": quantile(split_observed, 0.75),
        "split_profile_error_p75_ms": quantile(split_prediction_errors, 0.75),
        "p_profile_error_p75_ms": quantile(p_prediction_errors, 0.75),
        "combined_e_plus_p_profile_error_p75_ms": quantile(combined_prediction_errors, 0.75),
        "trace_image_order_valid_count": sum(1 for x in image_metrics if x["trace_order_valid"]),
        "trace_image_order_checked": sum(1 for x in image_metrics if x["request_id"]),
        "trace_first_token_id_match_count": sum(1 for x in image_metrics if x["first_token_id_match"]),
        "policy_decision_count": 0,
        "policy_selected_n": 0,
        "policy_selected_s": 0,
        "policy_split_eligible": 0,
        "policy_actual_split_count": 0,
        "sensitivity": sensitivity,
        "server_async_meta": None,
        "scheduler_counters": None,
        "missing_client_image_responses": max(0, len(image_sends) - len(image_responses)),
        "measurement_text_tokens": sum(token_counts.values()),
    }

    if row["policy"] == "A" and decision_rows:
        decision_by_id = {
            resolved: decision
            for decision in decision_rows
            if (resolved := resolve_request_id(str(decision.get("request_id", "")), image_ids)) is not None
        }
        decision_ids = set(decision_by_id)
        actual_split = {
            str(x.get("request_id")) for x in image_metrics
            if x.get("encoder_step_ns") is not None and x.get("prefill_step_ns") is not None
            and int(x["prefill_step_ns"]) > int(x["encoder_step_ns"])
        }
        metric["policy_decision_count"] = len(decision_rows)
        metric["policy_selected_n"] = sum(x.get("selected_action") == "N" for x in decision_rows)
        metric["policy_selected_s"] = sum(x.get("selected_action") == "S" for x in decision_rows)
        metric["policy_split_eligible"] = sum(bool(x.get("eligible_for_split")) for x in decision_rows)
        metric["policy_actual_split_count"] = len(actual_split & decision_ids)
        for image in image_metrics:
            decision = decision_by_id.get(str(image.get("request_id")))
            if decision is not None:
                image["selected_action"] = decision.get("selected_action")
                image["eligible_for_split"] = decision.get("eligible_for_split")
                image["selected_action_reason"] = decision.get("reason")
                image["d_slack_n_ms"] = decision.get("d_slack_n_ms")
                image["d_slack_s_ms"] = decision.get("d_slack_s_ms")
                image["i_slack_n_ms"] = decision.get("i_slack_n_ms")
                image["i_slack_s_ms"] = decision.get("i_slack_s_ms")
                image["e_hat_ms"] = decision.get("e_hat_ms")
                image["p_hat_ms"] = decision.get("p_hat_ms")
                image["decode_step_hat_ms"] = decision.get("decode_step_hat_ms")
                image["split_gap_hat_ms"] = decision.get("split_gap_hat_ms")

    async_meta_path = server_path.with_suffix(server_path.suffix + ".async-meta.json")
    if async_meta_path.exists():
        metric["server_async_meta"] = read_json(async_meta_path)
    scheduler_meta_path = scheduler_path.with_suffix(scheduler_path.suffix + ".async-meta.json")
    if scheduler_meta_path.exists():
        metric["scheduler_counters"] = read_json(scheduler_meta_path)

    details = {
        "image_requests": image_metrics,
        "sensitivity": sensitivity,
        "decision_path": str(decision_path) if decision_path.exists() else None,
        "decision_rows": decision_rows,
        "client_gaps_ms": client_gaps,
        "client_gap_intervals_by_lane": text_gap_by_lane,
        "scheduler_decode_gaps_ms": decode_gap_values,
        "observed_split_gap_ms": split_observed,
        "server_event_counts": server_counts,
    }
    return metric, details


def bootstrap_mean_ci(values: list[float], reps: int, rng: random.Random) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    estimate = statistics.fmean(values)
    if len(values) == 1:
        return estimate, estimate, estimate
    means = []
    for _ in range(reps):
        means.append(statistics.fmean(rng.choice(values) for _ in values))
    return estimate, quantile(means, 0.025), quantile(means, 0.975)


def summarize_runs(
    runs: list[dict[str, Any]],
    reps: int,
    analysis_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[int, int, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in runs:
        groups[(row["grid"], row["text_streams"], row["arrival_mode"], row["policy"])].append(row)
    cell_summary: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    metrics = [
        "text_max_gap_ms",
        "text_gap_p95_ms",
        "image_ttft_p95_ms",
        "joint_30_55_goodput_per_s",
        "image_queue_wait_p95_ms",
        "image_completion_rate",
        "text_completion_rate",
    ]
    diagnostics = [
        "scheduler_decode_gap_p75_ms",
        "p_profile_error_p75_ms",
        "combined_e_plus_p_profile_error_p75_ms",
        "split_profile_error_p75_ms",
        "formal_visual_forward_cpu_elapsed_p75_ms",
        "e_to_p_scheduler_gap_p95_ms",
    ]
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda x: x["seed"])
        row = {
            "grid": key[0], "text_streams": key[1],
            "arrival_mode": key[2], "policy": key[3],
            "n_seed_runs": len(group),
            "metrics_across_seed_runs": {},
            "sensitivity_across_seed_runs": {},
            "image_responses": sum(int(x["planned_image_requests"]) for x in group),
            "image_completed": sum(int(x["image_completed"]) for x in group),
            "image_errors": sum(int(x["image_errors"]) for x in group),
            "text_requests_started": sum(int(x["text_requests_started_in_measurement"]) for x in group),
            "text_completed": sum(int(x["text_completed"]) for x in group),
        }
        for metric in metrics + diagnostics:
            values = [float(x[metric]) for x in group if x.get(metric) is not None]
            row["metrics_across_seed_runs"][metric] = {
                "mean": mean(values),
                "p50": quantile(values, 0.50),
                "p95": quantile(values, 0.95),
                "p99": quantile(values, 0.99),
                "n": len(values),
            }
        for threshold in ("25/45", "30/55", "40/75"):
            values = [
                float(x["sensitivity"][threshold]["joint_goodput_per_s"])
                for x in group if threshold in x.get("sensitivity", {})
            ]
            row["sensitivity_across_seed_runs"][threshold] = {
                "joint_goodput_per_s_mean": mean(values),
                "joint_pass_rate_mean": mean([
                    float(x["sensitivity"][threshold]["joint_pass_rate"])
                    for x in group if x["sensitivity"][threshold]["joint_pass_rate"] is not None
                ]),
            }
        cell_summary.append(row)

    by_pair: dict[tuple[int, int, str, int], dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for row in runs:
        by_pair[(row["grid"], row["text_streams"], row["arrival_mode"], row["seed"])][row["policy"]] = row
    for condition in sorted({(x["grid"], x["text_streams"], x["arrival_mode"]) for x in runs}):
        policies = ["F", "S", "A"] + (["U"] if condition[2] == "burst" else [])
        for policy in policies:
            for metric in metrics:
                differences = []
                paired_seeds = []
                for seed in range(1000, 1012):
                    pair = by_pair.get((*condition, seed), {})
                    if "N" in pair and policy in pair and pair["N"].get(metric) is not None and pair[policy].get(metric) is not None:
                        differences.append(float(pair[policy][metric]) - float(pair["N"][metric]))
                        paired_seeds.append(seed)
                rng = random.Random(analysis_seed + condition[0] * 100 + condition[1] * 10 + len(condition[2]) + ord(policy[0]))
                estimate, low, high = bootstrap_mean_ci(differences, reps, rng)
                paired.append(
                    {
                        "grid": condition[0], "text_streams": condition[1],
                        "arrival_mode": condition[2], "policy_vs_N": policy,
                        "metric": metric, "paired_n": len(differences),
                        "seeds": paired_seeds,
                        "mean_paired_difference_candidate_minus_N": estimate,
                        "bootstrap_95_ci": [low, high],
                        "bootstrap_reps": reps,
                    }
                )
    return cell_summary, paired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--analysis-seed", type=int, default=20261107)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    schedule = read_json(args.schedule)
    profile = read_json(args.profile)
    schedule_copy = dict(schedule)
    recorded_schedule_sha = schedule_copy.pop("schedule_sha256", None)
    schedule_copy.pop("policy_cell_order", None)
    computed_schedule_sha = hashlib.sha256(
        json.dumps(schedule_copy, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    dataset_sha = hashlib.sha256(args.dataset_manifest.read_bytes()).hexdigest()
    profile_sha = hashlib.sha256(args.profile.read_bytes()).hexdigest()
    if profile.get("frozen") is not True:
        raise RuntimeError("estimator profile is not frozen")
    profile_sidecar = args.profile.with_suffix(args.profile.suffix + ".sha256")
    if not profile_sidecar.exists() or profile_sidecar.read_text(encoding="utf-8").split()[0] != profile_sha:
        raise RuntimeError("frozen estimator profile SHA-256 sidecar does not match")
    if recorded_schedule_sha != computed_schedule_sha:
        raise RuntimeError("formal schedule SHA-256 is invalid")
    if schedule.get("plan_revision") != "v4.3" or len(schedule.get("rows", [])) != 432:
        raise RuntimeError("formal schedule is not the frozen v4.3 432-run schedule")
    if schedule.get("estimator_profile_sha256") != profile_sha:
        raise RuntimeError("formal schedule profile hash does not match the frozen profile")
    if schedule.get("dataset_manifest_sha256") != dataset_sha:
        raise RuntimeError("formal schedule dataset manifest hash does not match")
    if manifest.get("matrix", {}).get("schedule_sha256") != schedule.get("schedule_sha256"):
        raise RuntimeError("manifest and formal schedule hashes differ")
    manifest_rows = manifest.get("matrix", {}).get("seed_records", [])
    done_by_key: dict[tuple[int, int, str, str, int], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for row in manifest_rows:
        key = (
            int(row["image_size"]), int(row["text_streams"]), str(row["arrival_mode"]),
            str(row["policy"]), int(row["seed"]),
        )
        if row.get("status") == "completed" and int(row.get("process_exit", 1)) == 0:
            done_by_key[key] = row
        else:
            failures.append({k: row.get(k) for k in ("run_id", "policy", "image_size", "text_streams", "arrival_mode", "seed", "attempt", "status", "process_exit", "error")})
    expected = {
        (int(r["grid"]), int(r["lanes"]), str(r["arrival"]), str(r["policy"]), int(r["seed"]))
        for r in schedule["rows"]
    }
    missing = sorted(expected - set(done_by_key))
    if missing:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        partial = {
            "status": "incomplete",
            "completed_unique_runs": len(done_by_key),
            "planned_runs": len(expected),
            "missing_keys": [list(x) for x in missing],
            "failed_attempts": failures,
            "note": "No final statistics are calculated until all registered seed-policy runs have a successful attempt.",
        }
        (args.output_dir / "formal-matrix-progress.json").write_text(json.dumps(partial, ensure_ascii=False, indent=2) + "\n")
        raise RuntimeError(f"formal matrix incomplete: {len(done_by_key)}/{len(expected)} successful keys")

    thresholds = ((25.0, 45.0), (30.0, 55.0), (40.0, 75.0))
    run_metrics: list[dict[str, Any]] = []
    request_details: list[dict[str, Any]] = []
    for schedule_row in schedule["rows"]:
        key = (
            int(schedule_row["grid"]), int(schedule_row["lanes"]),
            str(schedule_row["arrival"]), str(schedule_row["policy"]), int(schedule_row["seed"]),
        )
        indexed = done_by_key[key]
        metric, details = run_analysis(indexed, profile, thresholds)
        run_metrics.append(metric)
        image_by_id = {
            str(x.get("request_id")): x for x in details["image_requests"] if x.get("request_id")
        }
        for decision in details["decision_rows"]:
            image = image_by_id.get(str(decision.get("request_id")))
            if image is not None:
                decision["observed_ttft_ms"] = image.get("ttft_ms")
                decision["observed_joint_30_55_pass"] = image.get("joint_30_55_pass")
                decision["observed_actual_split"] = bool(
                    image.get("encoder_step_ns") is not None
                    and image.get("prefill_step_ns") is not None
                    and int(image["prefill_step_ns"]) > int(image["encoder_step_ns"])
                )
                decision["trace_order_valid"] = image.get("trace_order_valid")
        request_details.append(
            {
                "run_id": metric["run_id"],
                "policy": metric["policy"],
                "grid": metric["grid"],
                "text_streams": metric["text_streams"],
                "arrival_mode": metric["arrival_mode"],
                "seed": metric["seed"],
                "image_requests": details["image_requests"],
                "decision_rows": details["decision_rows"],
            }
        )

    cell_summary, paired = summarize_runs(run_metrics, args.bootstrap_reps, args.analysis_seed)
    decisions = [
        d for run in request_details for d in run["decision_rows"]
    ]
    bins: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    ratio_values = [float(x["e_hat_ms"]) / max(1e-9, float(x["p_hat_ms"])) for x in decisions if x.get("e_hat_ms") is not None and x.get("p_hat_ms") is not None]
    ratio_cut = quantile(ratio_values, 0.50)
    for decision in decisions:
        d_slack = decision.get("d_slack_n_ms")
        i_slack = decision.get("i_slack_n_ms")
        d_bin = "D-feasible" if d_slack is not None and float(d_slack) >= 0 else "D-violated-or-unobserved"
        i_bin = "I-feasible" if i_slack is not None and float(i_slack) >= 0 else "I-violated"
        ratio = float(decision.get("e_hat_ms", 0.0)) / max(1e-9, float(decision.get("p_hat_ms", 0.0)))
        r_bin = "E/P-low" if ratio <= (ratio_cut or 0.0) else "E/P-high"
        bins[(d_bin, i_bin, r_bin)].append(decision)
    state_partition = []
    for key, rows in sorted(bins.items()):
        state_partition.append(
            {
                "d_slack_bin": key[0], "i_slack_bin": key[1], "e_over_p_bin": key[2],
                "decisions": len(rows),
                "selected_N": sum(x.get("selected_action") == "N" for x in rows),
                "selected_S": sum(x.get("selected_action") == "S" for x in rows),
                "split_eligible": sum(bool(x.get("eligible_for_split")) for x in rows),
                "actual_split": sum(bool(x.get("observed_actual_split")) for x in rows),
                "joint_30_55_passes": sum(bool(x.get("observed_joint_30_55_pass")) for x in rows),
                "trace_order_valid": sum(bool(x.get("trace_order_valid")) for x in rows),
                "reasons": dict(collections.Counter(str(x.get("reason")) for x in rows)),
                "mean_d_slack_n_ms": mean([float(x["d_slack_n_ms"]) for x in rows if x.get("d_slack_n_ms") is not None]),
                "mean_d_slack_s_ms": mean([float(x["d_slack_s_ms"]) for x in rows if x.get("d_slack_s_ms") is not None]),
                "mean_i_slack_n_ms": mean([float(x["i_slack_n_ms"]) for x in rows if x.get("i_slack_n_ms") is not None]),
                "mean_i_slack_s_ms": mean([float(x["i_slack_s_ms"]) for x in rows if x.get("i_slack_s_ms") is not None]),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "formal-run-metrics.json").write_text(json.dumps(run_metrics, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "formal-request-decisions-and-images.json").write_text(json.dumps(request_details, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "formal-cell-summary.json").write_text(json.dumps(cell_summary, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "formal-paired-bootstrap.json").write_text(json.dumps(paired, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "formal-state-partition.json").write_text(json.dumps(state_partition, ensure_ascii=False, indent=2) + "\n")

    report = {
        "status": "complete",
        "schedule_sha256": schedule.get("schedule_sha256"),
        "profile_sha256": profile_sha,
        "planned_runs": len(expected),
        "successful_unique_runs": len(done_by_key),
        "failed_attempts_preserved": failures,
        "analysis_seed": args.analysis_seed,
        "bootstrap_reps": args.bootstrap_reps,
        "thresholds_text_gap_ms_image_ttft_ms": [list(x) for x in thresholds],
        "cell_count": len(cell_summary),
        "paired_comparison_count": len(paired),
        "a_decision_count": len(decisions),
        "request_trace_order_valid": sum(int(x["trace_image_order_valid_count"]) for x in run_metrics),
        "request_trace_order_checked": sum(int(x["trace_image_order_checked"]) for x in run_metrics),
        "first_token_id_matches": sum(int(x["trace_first_token_id_match_count"]) for x in run_metrics),
        "server_event_rows_written": sum(int((x.get("server_async_meta") or {}).get("written", 0)) for x in run_metrics),
        "server_event_rows_dropped": sum(int((x.get("server_async_meta") or {}).get("dropped", 0)) for x in run_metrics),
        "server_event_write_errors": sum(int((x.get("server_async_meta") or {}).get("write_errors", 0)) for x in run_metrics),
        "state_partition_bins": state_partition,
        "limitations": [
            "Trace seed is the independent unit for paired bootstrap intervals; requests and tokens are not treated as independent samples.",
            "Joint SLO goodput is counted per measurement image: image TTFT must meet the image deadline and every configured text lane must have complete interval coverage with its overlapping maximum token gap within the text deadline.",
            "Formal first-token engine-ready minus P-step can benchmark E+P in unsplit arms or P after E in split arms; formal mode does not collect CUDA-event visual durations, so E itself is not isolated on holdout.",
            "The formal visual-forward CPU elapsed time is a dispatch-side diagnostic proxy and is not interpreted as GPU execution latency.",
            "Missing client records and async writer metadata are reported per run; raw attempt evidence remains authoritative.",
        ],
    }
    (args.output_dir / "formal-analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    def fmt(value: Any) -> str:
        return "NA" if value is None else f"{float(value):.3f}"

    lines = [
        "# D 正式矩阵分析",
        "",
        f"- 有效 seed-policy runs：{len(done_by_key)}/432；策略条件：{len(cell_summary)}。",
        f"- Schedule SHA-256：{schedule.get('schedule_sha256')}。",
        f"- Frozen profile SHA-256：{report['profile_sha256']}。",
        f"- Bootstrap：以配对 trace seed 为单位，{args.bootstrap_reps} 次重采样；分析随机种子 {args.analysis_seed}。",
        "- 每个策略条件的 P50/P95/P99 指标先在 run 内计算，再以 12 个 seed 的 run-level 分布汇总；配对区间比较候选策略减 N。",
        "- 联合 SLO goodput：按测量期图像请求计数；图文 TTFT 达标，且各文本 lane 在该图像首 token 等待区间内的最大 client token gap 达标；报告每秒通过图像数与通过比例。",
        "- formal 模式没有 CUDA-event 视觉时长；E 单项留出误差不可直接观测。报告 engine-ready−P-step 与合并 E+P 估计误差，split 策略另报 P 与拆步间隔误差。",
        "",
        "## 各条件结果",
        "",
        "| Grid | 文本流 | 到达 | 策略 | Seed runs | 文本最大间隔 P95 (ms) | 图文 TTFT P95 中位值 (ms) | 联合 goodput (req/s) | 图像完成率 |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in cell_summary:
        metrics_for_cell = item["metrics_across_seed_runs"]
        lines.append(
            "| {grid} | {lanes} | {arrival} | {policy} | {n} | {text_gap} | {ttft} | {goodput} | {completion} |".format(
                grid=item["grid"], lanes=item["text_streams"], arrival=item["arrival_mode"],
                policy=item["policy"], n=item["n_seed_runs"],
                text_gap=fmt(metrics_for_cell["text_max_gap_ms"]["p95"]),
                ttft=fmt(metrics_for_cell["image_ttft_p95_ms"]["p50"]),
                goodput=fmt(metrics_for_cell["joint_30_55_goodput_per_s"]["mean"]),
                completion=fmt(metrics_for_cell["image_completion_rate"]["mean"]),
            )
        )
    lines.extend([
        "",
        "## 配对差值区间",
        "",
        "| Grid | 文本流 | 到达 | 候选减 N | 指标 | 均值差 | Bootstrap 95% CI | n |",
        "|---:|---:|---|---|---|---:|---:|---:|",
    ])
    for item in paired:
        ci = item["bootstrap_95_ci"]
        ci_text = f"[{fmt(ci[0])}, {fmt(ci[1])}]"
        lines.append(
            f"| {item['grid']} | {item['text_streams']} | {item['arrival_mode']} | {item['policy_vs_N']} | {item['metric']} | {fmt(item['mean_paired_difference_candidate_minus_N'])} | {ci_text} | {item['paired_n']} |"
        )
    lines.extend(["", "## Trace 完整性", "", f"- 首 token server/API/client 顺序通过：{report['request_trace_order_valid']}/{report['request_trace_order_checked']} 个已关联图像请求。", f"- 首 token ID 一致：{report['first_token_id_matches']} 个请求。", f"- server async writer 写入 {report['server_event_rows_written']} 行；丢弃 {report['server_event_rows_dropped']} 行；写错误 {report['server_event_write_errors']}。", f"- 被补跑前的失败 attempt 数：{len(failures)}，保留在 JSON 报告。", "", "## A 策略状态分区", "", "见 formal-state-partition.json；按 D slack 符号、I slack 符号、E/P 比值中位数分箱，并给出选择、实际拆步与联合 SLO 结果。", "", "## 限制", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    (args.output_dir / "formal-analysis.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "complete", "runs": len(done_by_key), "cells": len(cell_summary), "paired_comparisons": len(paired), "a_decisions": len(decisions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

