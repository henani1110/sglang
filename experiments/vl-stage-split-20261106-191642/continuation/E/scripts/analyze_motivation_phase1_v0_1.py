#!/usr/bin/env python3
"""Summarize Phase 1 paired causal-diagnosis traces without changing raw runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any


def quantile_nearest_rank(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def paired_burst_offsets(duration_s: int) -> list[float]:
    return [float(t) + delta for t in range(0, duration_s, 2) for delta in (0.0, 0.1)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_run(run_dir: Path) -> dict[str, Any]:
    result_path = run_dir / "result.json"
    events_path = run_dir / "client-events.jsonl"
    manifest_path = run_dir / "run-manifest.json"
    if not all(path.is_file() for path in (result_path, events_path, manifest_path)):
        raise FileNotFoundError(f"run is missing result, events, or manifest: {run_dir}")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cell_grid = int(manifest.get("cell", {}).get("grid", 0))
    source_grid = int(manifest.get("image_set", {}).get("size", 0))
    workload_grid_matches_cell = cell_grid > 0 and source_grid == cell_grid
    rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line]
    measurement_start = int(result["measurement_start_ns"])
    measurement_end = measurement_start + int(result["measurement_seconds"] * 1e9)
    warmup_starts = [
        int(row["time_ns"])
        for row in rows
        if row.get("kind") == "phase_start" and row.get("phase") == "warmup"
    ]
    warmup_start_ns = min(warmup_starts) if warmup_starts else None
    warmup_actual_s = (
        (measurement_start - warmup_start_ns) / 1e9 if warmup_start_ns is not None else None
    )
    warmup_drain_actual_s = next(
        (float(row["actual_duration_ms"]) / 1000.0 for row in rows if row.get("kind") == "warmup_drain_complete"),
        None,
    )
    measurement_ends = [
        int(row["time_ns"])
        for row in rows
        if row.get("kind") == "phase_end" and row.get("phase") == "measurement"
    ]
    measurement_actual_s = (max(measurement_ends) - measurement_start) / 1e9 if measurement_ends else None

    image_rows = [
        row
        for row in rows
        if row.get("kind") == "image_response"
        and row.get("phase") == "measurement"
        and row.get("planned_ns") is not None
        and measurement_start <= int(row["planned_ns"]) < measurement_end
    ]
    image_ttft_ms = [
        (int(row["first_token_ns"]) - int(row["started_ns"])) / 1e6
        for row in image_rows
        if row.get("first_token_ns") is not None and not row.get("error")
    ]

    lane_timestamps: dict[int, list[int]] = {}
    lane_response_rows: dict[int, list[tuple[int, list[int]]]] = {}
    within_request_chunk_gaps_ms: list[float] = []
    text_ttft_started_cohort_ms: list[float] = []
    text_ttft_first_token_window_ms: list[float] = []
    text_requests_started_in_window = 0
    text_started_cohort_missing_first_token = 0
    text_started_cohort_errors = 0
    text_output_tokens_in_window = 0
    text_request_completions_in_window = 0
    for row in rows:
        if row.get("kind") != "text_response":
            continue
        lane = row.get("lane")
        if lane is None:
            continue
        lane = int(lane)
        sequence = int(row.get("sequence", -1))
        in_window = [
            int(value)
            for value in row.get("token_times_ns", [])
            if measurement_start <= int(value) < measurement_end
        ]
        if in_window:
            text_output_tokens_in_window += len(in_window)
            lane_response_rows.setdefault(lane, []).append((sequence, in_window))
            raw_chunk_times = row.get("chunk_times_ns") or []
            if raw_chunk_times:
                chunk_times = [
                    int(timestamp)
                    for timestamp in raw_chunk_times
                    if measurement_start <= int(timestamp) < measurement_end
                ]
            else:
                chunk_times = []
                for timestamp in in_window:
                    if not chunk_times or timestamp != chunk_times[-1]:
                        chunk_times.append(timestamp)
            within_request_chunk_gaps_ms.extend(
                (later - earlier) / 1e6
                for earlier, later in zip(chunk_times, chunk_times[1:])
            )
        started_ns = row.get("started_ns")
        if started_ns is not None and measurement_start <= int(started_ns) < measurement_end:
            text_requests_started_in_window += 1
            if row.get("error"):
                text_started_cohort_errors += 1
            first_token_ns = row.get("first_token_ns")
            if first_token_ns is None:
                text_started_cohort_missing_first_token += 1
            else:
                text_ttft_started_cohort_ms.append((int(first_token_ns) - int(started_ns)) / 1e6)
                if measurement_start <= int(first_token_ns) < measurement_end:
                    text_ttft_first_token_window_ms.append((int(first_token_ns) - int(started_ns)) / 1e6)
        if row.get("finished_ns") is not None and measurement_start <= int(row["finished_ns"]) < measurement_end:
            text_request_completions_in_window += 1
        if not row.get("error"):
            timestamps = lane_timestamps.setdefault(int(lane), [])
            timestamps.extend(in_window)

    cross_request_handoff_gaps_ms: list[float] = []
    for lane_rows in lane_response_rows.values():
        lane_rows.sort(key=lambda item: item[0])
        for (previous_sequence, previous_times), (next_sequence, next_times) in zip(lane_rows, lane_rows[1:]):
            if next_sequence != previous_sequence + 1:
                continue
            gap_ns = next_times[0] - previous_times[-1]
            if gap_ns >= 0:
                cross_request_handoff_gaps_ms.append(gap_ns / 1e6)

    def distribution(values: list[float]) -> dict[str, Any]:
        return {
            "observations": len(values),
            "p50": statistics.median(values) if values else None,
            "p95": quantile_nearest_rank(values, 0.95),
            "p99": quantile_nearest_rank(values, 0.99),
            "max": max(values) if values else None,
        }

    lane_metrics: dict[str, dict[str, Any]] = {}
    for lane, timestamps in sorted(lane_timestamps.items()):
        timestamps.sort()
        gaps_ms = [
            (later - earlier) / 1e6
            for earlier, later in zip(timestamps, timestamps[1:])
        ]
        lane_metrics[str(lane)] = {
            "token_timestamps": len(timestamps),
            "observed_gaps": len(gaps_ms),
            "max_gap_ms": max(gaps_ms) if gaps_ms else None,
            "p50_gap_ms": statistics.median(gaps_ms) if gaps_ms else None,
            "p95_gap_ms": quantile_nearest_rank(gaps_ms, 0.95),
            "p99_gap_ms": quantile_nearest_rank(gaps_ms, 0.99),
            "gap_compliance": {
                str(limit): (sum(gap <= limit for gap in gaps_ms) / len(gaps_ms) if gaps_ms else None)
                for limit in (25, 30, 40)
            },
        }

    lane_max_gaps = [
        lane["max_gap_ms"]
        for lane in lane_metrics.values()
        if lane["max_gap_ms"] is not None
    ]
    expected_lanes = int(manifest.get("cell", {}).get("active_text_lanes", 0))
    expected_text_tokens = int(result.get("text_tokens_per_request", 192))
    all_text_response_rows = [row for row in rows if row.get("kind") == "text_response"]
    stored_lane_integrity = result.get("text_lane_integrity", {})
    text_lane_integrity: dict[str, dict[str, Any]] = {}
    for lane in range(expected_lanes):
        lane_rows = [row for row in all_text_response_rows if row.get("lane") == lane]
        sequences = sorted(
            int(row["sequence"]) for row in lane_rows if row.get("sequence") is not None
        )
        stored = stored_lane_integrity.get(str(lane), {}) if isinstance(stored_lane_integrity, dict) else {}
        requests_started = int(stored.get("requests_started", len(lane_rows)))
        measured_tokens = sum(
            1 for row in lane_rows
            for timestamp in (row.get("token_times_ns") or [])
            if measurement_start <= int(timestamp) < measurement_end
        )
        complete_responses = sum(
            not row.get("error")
            and len(row.get("token_ids") or []) == expected_text_tokens
            and len(row.get("token_times_ns") or []) == len(row.get("token_ids") or [])
            for row in lane_rows
        )
        sequence_complete = sequences == list(range(requests_started))
        lane_valid = (
            requests_started > 0
            and len(lane_rows) == requests_started
            and sequence_complete
            and complete_responses == len(lane_rows)
            and measured_tokens > 0
        )
        text_lane_integrity[str(lane)] = {
            "requests_started": requests_started,
            "responses_recorded": len(lane_rows),
            "response_sequences": sequences,
            "sequence_complete": sequence_complete,
            "complete_responses": complete_responses,
            "measurement_tokens": measured_tokens,
            "valid": lane_valid,
        }
    raw_text_task_failures = sum(row.get("kind") == "text_lane_task_failure" for row in rows)
    text_task_failure_count = max(
        int(result.get("text_task_failure_count", 0)), raw_text_task_failures
    )
    text_integrity_recomputed = (
        expected_lanes > 0
        and set(text_lane_integrity) == {str(lane) for lane in range(expected_lanes)}
        and all(row["valid"] for row in text_lane_integrity.values())
        and text_task_failure_count == 0
        and not bool(result.get("text_task_timed_out", False))
    )
    causal_diagnostic_protocol = manifest.get("causal_diagnostic_protocol") == "motivation-phase1-v0.1"
    text_integrity_gate = (
        not causal_diagnostic_protocol
        or (text_integrity_recomputed and result.get("text_lane_integrity_valid") is True)
    )
    text_slo_pass_by_limit = {
        str(limit): (
            expected_lanes > 0
            and len(lane_metrics) == expected_lanes
            and len(lane_max_gaps) == expected_lanes
            and all(gap <= float(limit) for gap in lane_max_gaps)
        )
        for limit in (25, 30, 40)
    }
    text_slo_pass = text_slo_pass_by_limit["30"]
    image_slo_counts = {str(limit): sum(value <= limit for value in image_ttft_ms) for limit in (45, 55, 75)}
    planned_images = int(result.get("measurement_images_planned", len(image_rows)))
    successful_image_count = sum(not row.get("error") and row.get("first_token_ns") is not None for row in image_rows)
    arrival_mode = str(manifest.get("cell", {}).get("arrival_mode", result.get("arrival_mode", "")))
    placebo_events = [
        row for row in rows
        if row.get("kind") == "image_placebo_anchor"
        and row.get("phase") == "measurement"
        and row.get("planned_ns") is not None
        and measurement_start <= int(row["planned_ns"]) < measurement_end
    ]
    placebo_planned = int(result.get("measurement_placebo_anchors_planned", 0))
    placebo_recorded = int(result.get("measurement_placebo_anchors_recorded", 0))
    all_placebo_events = [row for row in rows if row.get("kind") == "image_placebo_anchor"]
    warmup_seconds = int(result.get("warmup_seconds", 60))
    measurement_seconds = int(result.get("measurement_seconds", 180))
    expected_warmup_offsets = paired_burst_offsets(warmup_seconds)
    expected_measurement_offsets = paired_burst_offsets(measurement_seconds)
    placebo_actual_drift_ms = [
        (int(row["time_ns"]) - int(row["planned_ns"])) / 1e6
        for row in all_placebo_events
        if row.get("time_ns") is not None and row.get("planned_ns") is not None
    ]
    image_send_rows = [row for row in rows if row.get("kind") == "image_request_send"]
    image_actual_drift_ms = [
        (int(row["actual_send_ns"]) - int(row["planned_ns"])) / 1e6
        for row in image_send_rows
        if row.get("actual_send_ns") is not None and row.get("planned_ns") is not None
    ]

    def phase_placebo_schedule_valid(
        phase: str, start_ns: int | None, offsets: list[float], first_anchor_index: int
    ) -> bool:
        phase_events = sorted(
            [row for row in all_placebo_events if row.get("phase") == phase],
            key=lambda row: int(row.get("anchor_index", -1)),
        )
        if start_ns is None or len(phase_events) != len(offsets):
            return False
        for index, (row, offset_s) in enumerate(zip(phase_events, offsets)):
            expected_ns = start_ns + int(offset_s * 1e9)
            if (
                row.get("planned_offset_s") != offset_s
                or row.get("planned_ns") != expected_ns
                or row.get("anchor_index") != first_anchor_index + index
                or row.get("burst_index") != index // 2
                or row.get("within_burst_index") != index % 2
                or row.get("placebo_for_arrival_mode") != "burst"
                or row.get("time_ns") is None
            ):
                return False
        return True

    placebo_schedule_valid = True
    if arrival_mode == "none":
        placebo_schedule_valid = (
            phase_placebo_schedule_valid("warmup", warmup_start_ns, expected_warmup_offsets, 0)
            and phase_placebo_schedule_valid(
                "measurement", measurement_start, expected_measurement_offsets, len(expected_warmup_offsets)
            )
            and len(all_placebo_events) == len(expected_warmup_offsets) + len(expected_measurement_offsets)
            and int(result.get("warmup_placebo_anchors_planned", 0)) == len(expected_warmup_offsets)
            and int(result.get("warmup_placebo_anchors_recorded", 0)) == len(expected_warmup_offsets)
            and placebo_planned == len(expected_measurement_offsets)
            and placebo_recorded == len(expected_measurement_offsets)
        )
    image_accounting_valid = (
        planned_images == int(result.get("measurement_image_responses", 0))
        and planned_images == int(result.get("measurement_images_sent", 0))
        and int(result.get("measurement_image_errors", 0)) == 0
    )
    if arrival_mode == "none":
        image_http_events = sum(
            row.get("kind") in ("image_request_send", "image_response") for row in rows
        )
        image_accounting_valid = (
            image_accounting_valid
            and planned_images == 0
            and int(result.get("measurement_images_sent", 0)) == 0
            and int(result.get("image_requests", 0)) == 0
            and image_http_events == 0
            and placebo_schedule_valid
            and len(placebo_events) == placebo_recorded
        )
    image_schedule_valid = True
    if arrival_mode == "burst":
        def phase_image_schedule_valid(
            phase: str, start_ns: int | None, offsets: list[float], first_slot: int
        ) -> bool:
            phase_events = sorted(
                [row for row in image_send_rows if row.get("phase") == phase],
                key=lambda row: int(row.get("image_slot", -1)),
            )
            if start_ns is None or len(phase_events) != len(offsets):
                return False
            for index, (row, offset_s) in enumerate(zip(phase_events, offsets)):
                if (
                    row.get("image_slot") != first_slot + index
                    or row.get("planned_offset_s") != offset_s
                    or row.get("planned_ns") != start_ns + int(offset_s * 1e9)
                    or row.get("actual_send_ns") is None
                ):
                    return False
            return len({int(row["image_slot"]) for row in phase_events}) == len(offsets)

        image_schedule_valid = (
            phase_image_schedule_valid("warmup", warmup_start_ns, expected_warmup_offsets, 0)
            and phase_image_schedule_valid(
                "measurement", measurement_start, expected_measurement_offsets, len(expected_warmup_offsets)
            )
            and len(image_send_rows) == len(expected_warmup_offsets) + len(expected_measurement_offsets)
        )
        warmup_image_responses = [
            row for row in rows
            if row.get("kind") == "image_response" and row.get("phase") == "warmup"
        ]
        measurement_image_responses = [
            row for row in rows
            if row.get("kind") == "image_response" and row.get("phase") == "measurement"
        ]
        image_accounting_valid = (
            image_accounting_valid
            and image_schedule_valid
            and len(warmup_image_responses) == len(expected_warmup_offsets)
            and len(measurement_image_responses) == len(expected_measurement_offsets)
            and all(row.get("first_token_ns") is not None and not row.get("error") for row in warmup_image_responses)
            and all(row.get("first_token_ns") is not None and not row.get("error") for row in measurement_image_responses)
            and int(result.get("image_requests", 0)) == len(warmup_image_responses) + len(measurement_image_responses)
            and int(result.get("image_errors", 0)) == 0
        )

    local_anchor_events: list[tuple[int, int]] = []
    if arrival_mode == "none":
        local_anchor_events = [
            (int(row["burst_index"]), int(row["planned_ns"]))
            for row in placebo_events
            if row.get("burst_index") is not None and row.get("within_burst_index") == 0
        ]
    elif arrival_mode == "burst":
        grouped_image_anchors: dict[int, list[dict[str, Any]]] = {}
        warmup_pair_count = len(expected_warmup_offsets) // 2
        for row in image_send_rows:
            if row.get("phase") != "measurement" or row.get("image_slot") is None:
                continue
            pair_index = int(row["image_slot"]) // 2 - warmup_pair_count
            grouped_image_anchors.setdefault(pair_index, []).append(row)
        for pair_index, pair_rows in grouped_image_anchors.items():
            if len(pair_rows) == 2:
                local_anchor_events.append((pair_index, min(int(row["planned_ns"]) for row in pair_rows)))
    planned_anchor_pairs = len(expected_measurement_offsets) // 2
    local_anchor_events = [
        (pair_index, anchor_ns)
        for pair_index, anchor_ns in local_anchor_events
        if anchor_ns - 50_000_000 >= measurement_start
        and anchor_ns + 500_000_000 <= measurement_end
    ]

    active_anchor_gaps_ms: list[float] = []
    active_anchor_counts: list[int] = []
    anchor_window_rows: list[dict[str, Any]] = []
    text_response_rows = [row for row in rows if row.get("kind") == "text_response"]
    for pair_index, anchor_ns in sorted(local_anchor_events):
        window_start_ns = anchor_ns - 50_000_000
        window_end_ns = anchor_ns + 500_000_000
        active_requests = []
        window_gaps_ms: list[float] = []
        for row in text_response_rows:
            started_ns = row.get("started_ns")
            if started_ns is None or int(started_ns) >= anchor_ns:
                continue
            chunk_values = row.get("chunk_times_ns") or row.get("token_times_ns") or []
            chunk_times: list[int] = []
            for value in chunk_values:
                timestamp = int(value)
                if not chunk_times or timestamp != chunk_times[-1]:
                    chunk_times.append(timestamp)
            had_output_before_anchor = any(timestamp < anchor_ns for timestamp in chunk_times)
            finished_ns = row.get("finished_ns")
            active_at_anchor = finished_ns is None or int(finished_ns) >= anchor_ns
            if not (had_output_before_anchor and active_at_anchor):
                continue
            active_requests.append(row)
            for earlier, later in zip(chunk_times, chunk_times[1:]):
                if later > window_start_ns and earlier < window_end_ns:
                    window_gaps_ms.append((later - earlier) / 1e6)
        active_anchor_counts.append(len(active_requests))
        active_anchor_gaps_ms.extend(window_gaps_ms)
        anchor_window_rows.append({
            "burst_index": pair_index,
            "planned_first_anchor_ns": anchor_ns,
            "active_preanchor_requests": len(active_requests),
            "overlapping_request_internal_chunk_gaps": len(window_gaps_ms),
            "request_internal_chunk_gap_ms": distribution(window_gaps_ms),
        })

    anchor_local_summary = {
        "applicable": arrival_mode in ("none", "burst"),
        "window_relative_to_first_planned_anchor": "[-50ms, +500ms)",
        "planned_pairs_in_measurement": planned_anchor_pairs,
        "paired_windows": len(anchor_window_rows),
        "boundary_pairs_excluded": planned_anchor_pairs - len(anchor_window_rows),
        "windows_with_active_preanchor_requests": sum(count > 0 for count in active_anchor_counts),
        "active_preanchor_requests_per_window": distribution([float(value) for value in active_anchor_counts]),
        "request_internal_gaps_overlapping_windows_ms": distribution(active_anchor_gaps_ms),
        "windows": anchor_window_rows,
        "interpretation": "supportive timing description; run is the replicate, and windows inherit earlier treatment and queue history",
    }
    joint_good_images = image_slo_counts["55"] if text_slo_pass else 0
    joint_goodput_applicable = arrival_mode != "none"
    joint_sensitivity = {}
    for text_limit, image_limit in ((25, 45), (30, 55), (40, 75)):
        count = image_slo_counts[str(image_limit)] if text_slo_pass_by_limit[str(text_limit)] else 0
        joint_sensitivity[f"{text_limit}/{image_limit}"] = {
            "text_slo_pass": text_slo_pass_by_limit[str(text_limit)],
            "good_image_requests": count if joint_goodput_applicable else None,
            "images_per_second": count / float(result["measurement_seconds"]) if joint_goodput_applicable else None,
            "applicable": joint_goodput_applicable,
        }

    controller_rows = [
        row.get("controller_decision")
        for row in rows
        if row.get("kind") == "image_request_send" and row.get("phase") == "measurement"
    ]
    controller_rows = [row for row in controller_rows if isinstance(row, dict)]
    chosen_ratios = [float(row["chosen_keep_ratio"]) for row in controller_rows if row.get("chosen_keep_ratio") is not None]
    expected_tokens = [int(row["chosen_visual_tokens_expected"]) for row in controller_rows if row.get("chosen_visual_tokens_expected") is not None]
    fallback_counts: dict[str, int] = {}
    route_counts: dict[str, int] = {}
    for row in controller_rows:
        fallback = row.get("fallback_reason")
        route = row.get("content_route", {}).get("route") if isinstance(row.get("content_route"), dict) else None
        if fallback:
            fallback_counts[str(fallback)] = fallback_counts.get(str(fallback), 0) + 1
        if route:
            route_counts[str(route)] = route_counts.get(str(route), 0) + 1

    visual_summary_v02 = run_dir / "server-visual-forward-summary-v0.2.json"
    visual_summary_v01 = run_dir / "server-visual-forward-summary.json"
    manager_summary_path = run_dir / "manager-summary.json"
    manager_summary = (
        json.loads(manager_summary_path.read_text(encoding="utf-8"))
        if manager_summary_path.is_file()
        else None
    )
    manager_summary_hash_valid = bool(
        manager_summary is not None
        and manifest.get("manager_summary_sha256") == sha256(manager_summary_path)
    )
    manager_result_hash_valid = bool(
        manager_summary is not None
        and manager_summary.get("result_sha256") == sha256(result_path)
    )
    expected_cell = manifest.get("cell", {})
    eligibility_reasons: list[str] = []
    if manifest.get("causal_diagnostic_protocol") != "motivation-phase1-v0.1":
        eligibility_reasons.append("protocol_mismatch")
    if manifest.get("block_id") != "motivation-phase1-v0.1":
        eligibility_reasons.append("block_id_mismatch")
    if manifest.get("status") != "completed" or result.get("status") != "completed":
        eligibility_reasons.append("run_incomplete")
    if manifest.get("formal_online_run") is not True:
        eligibility_reasons.append("not_marked_formal_online_run")
    if result.get("seed") != manifest.get("seed"):
        eligibility_reasons.append("seed_mismatch")
    if manifest.get("arm") != "N" or result.get("policy") != "N" or manifest.get("server_mode") != "N":
        eligibility_reasons.append("not_native_N_arm")
    if (
        expected_cell.get("grid") != 768
        or expected_cell.get("active_text_lanes") not in (2, 6)
        or expected_cell.get("arrival_mode") not in ("none", "burst")
    ):
        eligibility_reasons.append("cell_mismatch")
    if (
        manifest.get("warmup_seconds") != 60
        or manifest.get("measurement_seconds") != 180
        or manifest.get("text_tokens_per_request") != 192
        or manifest.get("image_tokens_per_request") != 16
        or manifest.get("visual_keep_ratio") != 1.0
        or result.get("arrival_mode") != expected_cell.get("arrival_mode")
        or result.get("image_size") != expected_cell.get("grid")
        or result.get("text_streams") != expected_cell.get("active_text_lanes")
        or result.get("warmup_seconds") != manifest.get("warmup_seconds")
        or result.get("measurement_seconds") != manifest.get("measurement_seconds")
        or result.get("text_tokens_per_request") != manifest.get("text_tokens_per_request")
    ):
        eligibility_reasons.append("config_mismatch")
    if not all(manifest.get(key) for key in (
        "manager_script_sha256", "campaign_sha256", "server_script_sha256", "image_manifest_sha256"
    )):
        eligibility_reasons.append("missing_artifact_fingerprint")
    if manager_summary is None or manager_summary.get("experiment_arm_eligible") is not True:
        eligibility_reasons.append("manager_ineligible")
    if manager_summary is None or manager_summary.get("port_closed_after") is not True:
        eligibility_reasons.append("server_port_not_closed")
    if not manager_summary_hash_valid or not manager_result_hash_valid:
        eligibility_reasons.append("manager_artifact_hash_mismatch")
    if not image_accounting_valid:
        eligibility_reasons.append("image_schedule_or_accounting_invalid")
    if result.get("text_errors") != 0 or not text_integrity_gate:
        eligibility_reasons.append("text_lane_invalid")
    if (
        warmup_actual_s is None
        or warmup_actual_s < float(manifest.get("warmup_seconds", 60))
        or measurement_actual_s is None
        or measurement_actual_s < float(manifest.get("measurement_seconds", 180))
    ):
        eligibility_reasons.append("duration_invalid")
    if not workload_grid_matches_cell:
        eligibility_reasons.append("image_grid_mismatch")

    return {
        "run_id": result.get("run_id"),
        "arm": result.get("policy"),
        "block_id": manifest.get("block_id"),
        "protocol": manifest.get("causal_diagnostic_protocol"),
        "manifest_status": manifest.get("status"),
        "cell": manifest.get("cell"),
        "declared_input_grid_pixels": source_grid,
        "workload_grid_matches_cell": workload_grid_matches_cell,
        "seed": result.get("seed"),
        "status": result.get("status"),
        "formal_arm_eligible": not eligibility_reasons,
        "ineligibility_reasons": eligibility_reasons,
        "artifact_fingerprints": {
            key: manifest.get(key)
            for key in (
                "manager_script_sha256", "campaign_sha256", "server_script_sha256", "image_manifest_sha256"
            )
        },
        "experiment_config": {
            "warmup_seconds": manifest.get("warmup_seconds"),
            "measurement_seconds": manifest.get("measurement_seconds"),
            "text_tokens_per_request": manifest.get("text_tokens_per_request"),
            "image_tokens_per_request": manifest.get("image_tokens_per_request"),
            "model_path": manifest.get("model_path"),
            "server_mode": manifest.get("server_mode"),
            "visual_keep_ratio": manifest.get("visual_keep_ratio"),
        },
        "manager_eligibility": {
            "manager_experiment_arm_eligible": manager_summary.get("experiment_arm_eligible") if manager_summary else None,
            "port_closed_after": manager_summary.get("port_closed_after") if manager_summary else None,
            "manager_summary_hash_valid": manager_summary_hash_valid,
            "manager_result_hash_valid": manager_result_hash_valid,
        },
        "warmup_actual_seconds": warmup_actual_s,
        "warmup_drain_actual_seconds": warmup_drain_actual_s,
        "measurement_seconds": result.get("measurement_seconds"),
        "measurement_actual_seconds": measurement_actual_s,
        "image_accounting": {
            "arrival_mode": arrival_mode,
            "planned_burst_schedule_valid": image_schedule_valid,
            "image_send_actual_drift_ms": distribution(image_actual_drift_ms),
            "planned_image_slots": int(result.get("measurement_image_slots", planned_images)),
            "planned_image_requests": planned_images,
            "image_requests_sent": int(result.get("measurement_images_sent", 0)),
            "image_responses": int(result.get("measurement_image_responses", 0)),
            "placebo_anchors_planned": placebo_planned,
            "placebo_anchors_recorded": placebo_recorded,
            "placebo_anchor_events_in_window": len(placebo_events),
            "placebo_anchor_schedule_valid": placebo_schedule_valid,
            "placebo_anchor_actual_drift_ms": distribution(placebo_actual_drift_ms),
            "accounting_valid": image_accounting_valid,
        },
        "image": {
            "applicable": arrival_mode != "none",
            "planned": planned_images,
            "responses_with_first_token": successful_image_count,
            "errors": int(result.get("measurement_image_errors", 0)),
            "missing_or_no_first_token": max(0, planned_images - successful_image_count),
            "ttft_ms": {
                "p50": statistics.median(image_ttft_ms) if image_ttft_ms else None,
                "p95": quantile_nearest_rank(image_ttft_ms, 0.95),
                "p99": quantile_nearest_rank(image_ttft_ms, 0.99),
                "max": max(image_ttft_ms) if image_ttft_ms else None,
            },
            "ttft_le_ms": image_slo_counts,
        },
        "text": {
            "errors": int(result.get("text_errors", 0)),
            "task_failure_count": text_task_failure_count,
            "task_timed_out": bool(result.get("text_task_timed_out", False)),
            "integrity_recomputed_valid": text_integrity_recomputed,
            "integrity_campaign_valid": result.get("text_lane_integrity_valid"),
            "integrity_by_lane": text_lane_integrity,
            "requests_started_in_window": text_requests_started_in_window,
            "request_completions_in_window": text_request_completions_in_window,
            "started_cohort_missing_first_token": text_started_cohort_missing_first_token,
            "started_cohort_errors": text_started_cohort_errors,
            "output_tokens_in_window": text_output_tokens_in_window,
            "output_tokens_per_second": text_output_tokens_in_window / float(result["measurement_seconds"]),
            "ttft_started_cohort_ms": distribution(text_ttft_started_cohort_ms),
            "ttft_request_and_first_token_in_window_ms": distribution(text_ttft_first_token_window_ms),
            "lanes": lane_metrics,
            "client_visible_request_internal_chunk_gap_ms": distribution(within_request_chunk_gaps_ms),
            "client_visible_cross_request_handoff_gap_ms": distribution(cross_request_handoff_gaps_ms),
            "anchor_local_time_association": anchor_local_summary,
            "all_lanes_max_gap_le_30ms": text_slo_pass,
        },
        "joint_goodput": {
            "applicable": joint_goodput_applicable,
            "definition": (
                "not applicable for the no-image control"
                if not joint_goodput_applicable
                else "count of measurement image requests with TTFT <=55ms per measurement second, only when every text lane's in-window max token gap <=30ms; request errors/missing responses are zero-goodput"
            ),
            "good_image_requests": joint_good_images if joint_goodput_applicable else None,
            "images_per_second": joint_good_images / float(result["measurement_seconds"]) if joint_goodput_applicable else None,
        },
        "joint_goodput_sensitivity": joint_sensitivity,
        "candidate_policy_trace": {
            "image_decisions": len(controller_rows),
            "chosen_keep_ratio_values": chosen_ratios,
            "expected_retained_visual_tokens": expected_tokens,
            "content_route_counts": route_counts,
            "fallback_counts": fallback_counts,
        },
        "server_visual_forward_summary": json.loads(
            (visual_summary_v02 if visual_summary_v02.is_file() else visual_summary_v01).read_text(encoding="utf-8")
        ) if visual_summary_v02.is_file() or visual_summary_v01.is_file() else None,
        "input_hashes": {
            "client_events_sha256": sha256(events_path),
            "run_manifest_sha256": sha256(manifest_path),
            "result_sha256": sha256(result_path),
        },
    }


def incomplete_run_summary(run_dir: Path, error: Exception) -> dict[str, Any]:
    def read_json_if_present(name: str) -> dict[str, Any]:
        path = run_dir / name
        try:
            value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    result = read_json_if_present("result.json")
    manifest = read_json_if_present("run-manifest.json")
    name_match = re.fullmatch(r"E-N-G(?P<grid>\d+)-L(?P<lanes>\d+)-(?P<mode>none|burst)-s(?P<seed>\d+)-a(?P<attempt>\d+)", run_dir.name)
    parsed = name_match.groupdict() if name_match else {}
    inferred_identity = bool(parsed and run_dir.parent.name == "motivation-phase1-v0.1")
    cell = manifest.get("cell") or ({
        "grid": int(parsed["grid"]),
        "active_text_lanes": int(parsed["lanes"]),
        "arrival_mode": parsed["mode"],
    } if inferred_identity else {})
    inferred_block = "motivation-phase1-v0.1" if inferred_identity else None
    return {
        "run_id": result.get("run_id") or manifest.get("run_id") or run_dir.name,
        "arm": result.get("policy") or manifest.get("arm") or ("N" if inferred_identity else None),
        "block_id": manifest.get("block_id") or inferred_block,
        "protocol": manifest.get("causal_diagnostic_protocol") or inferred_block,
        "manifest_status": manifest.get("status"),
        "cell": cell,
        "seed": result.get("seed", manifest.get("seed", int(parsed["seed"]) if inferred_identity else None)),
        "status": result.get("status") or manifest.get("status") or "incomplete",
        "attempt": int(parsed["attempt"]) if inferred_identity else None,
        "identity_inferred_from_run_directory": inferred_identity,
        "formal_arm_eligible": False,
        "ineligibility_reasons": ["analysis_incomplete_result"],
        "artifact_fingerprints": {
            key: manifest.get(key)
            for key in (
                "manager_script_sha256", "campaign_sha256", "server_script_sha256", "image_manifest_sha256"
            )
        },
        "experiment_config": {
            "warmup_seconds": manifest.get("warmup_seconds"),
            "measurement_seconds": manifest.get("measurement_seconds"),
            "text_tokens_per_request": manifest.get("text_tokens_per_request"),
            "image_tokens_per_request": manifest.get("image_tokens_per_request"),
            "model_path": manifest.get("model_path"),
            "server_mode": manifest.get("server_mode"),
            "visual_keep_ratio": manifest.get("visual_keep_ratio"),
        },
        "analysis_error": f"{type(error).__name__}: {error}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", nargs="*", help="Optional exact run directory names")
    args = parser.parse_args()
    selected = [args.runs_root / name for name in args.runs] if args.runs else sorted(args.runs_root.iterdir())
    summaries = []
    for run_dir in selected:
        if not run_dir.is_dir():
            continue
        recognizable_empty_attempt = (
            run_dir.parent.name == "motivation-phase1-v0.1"
            and re.fullmatch(r"E-N-G\d+-L(?:2|6)-(?:none|burst)-s\d+-a\d+", run_dir.name) is not None
        )
        if (
            not (run_dir / "result.json").exists()
            and not (run_dir / "run-manifest.json").exists()
            and not recognizable_empty_attempt
        ):
            continue
        try:
            summaries.append(summarize_run(run_dir))
        except Exception as exc:
            summaries.append(incomplete_run_summary(run_dir, exc))

    metric_paths = {
        "request_internal_chunk_gap_p95_ms": ("text", "client_visible_request_internal_chunk_gap_ms", "p95"),
        "cross_request_handoff_gap_p95_ms": ("text", "client_visible_cross_request_handoff_gap_ms", "p95"),
        "text_ttft_started_cohort_p95_ms": ("text", "ttft_started_cohort_ms", "p95"),
        "text_output_tokens_per_second": ("text", "output_tokens_per_second"),
        "text_requests_started_in_window": ("text", "requests_started_in_window"),
    }

    def at_path(row: dict[str, Any], path: tuple[str, ...]) -> float | None:
        value: Any = row
        for key in path:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return None if value is None else float(value)

    phase1_attempts = [
        row for row in summaries
        if row.get("block_id") == "motivation-phase1-v0.1"
        or row.get("protocol") == "motivation-phase1-v0.1"
    ]
    expected_seeds = [41001, 41002, 41003, 41004, 41005, 41006]
    attempt_audit = []
    pair_audit = []
    valid_pairs: list[dict[str, Any]] = []
    for seed in expected_seeds:
        for lane_count in (2, 6):
            mode_attempts: dict[str, list[dict[str, Any]]] = {"none": [], "burst": []}
            for row in phase1_attempts:
                cell = row.get("cell") or {}
                if (
                    row.get("seed") == seed
                    and cell.get("grid") == 768
                    and cell.get("active_text_lanes") == lane_count
                    and cell.get("arrival_mode") in mode_attempts
                    and row.get("arm") == "N"
                ):
                    mode_attempts[str(cell["arrival_mode"])].append(row)
            for mode in ("none", "burst"):
                attempts = mode_attempts[mode]
                eligible = [row for row in attempts if row.get("formal_arm_eligible") is True]
                ineligibility_reason_counts: dict[str, int] = {}
                for row in attempts:
                    for reason in row.get("ineligibility_reasons", []):
                        ineligibility_reason_counts[str(reason)] = ineligibility_reason_counts.get(str(reason), 0) + 1
                attempt_audit.append({
                    "seed": seed,
                    "active_text_lanes": lane_count,
                    "arrival_mode": mode,
                    "attempt_count": len(attempts),
                    "eligible_attempt_count": len(eligible),
                    "ineligible_attempt_count": len(attempts) - len(eligible),
                    "ineligibility_reason_counts": ineligibility_reason_counts,
                    "attempts": [
                        {
                            "run_id": row.get("run_id"),
                            "status": row.get("status"),
                            "eligible": bool(row.get("formal_arm_eligible")),
                            "ineligibility_reasons": row.get("ineligibility_reasons", []),
                            "analysis_error": row.get("analysis_error"),
                        }
                        for row in attempts
                    ],
                })

            none_eligible = [row for row in mode_attempts["none"] if row.get("formal_arm_eligible") is True]
            burst_eligible = [row for row in mode_attempts["burst"] if row.get("formal_arm_eligible") is True]
            reasons = []
            if len(none_eligible) != 1:
                reasons.append("none_missing_or_duplicate_eligible_attempt")
            if len(burst_eligible) != 1:
                reasons.append("burst_missing_or_duplicate_eligible_attempt")
            if len(none_eligible) == 1 and len(burst_eligible) == 1:
                none_row, burst_row = none_eligible[0], burst_eligible[0]
                if none_row.get("artifact_fingerprints") != burst_row.get("artifact_fingerprints"):
                    reasons.append("paired_artifact_fingerprint_mismatch")
                if none_row.get("experiment_config") != burst_row.get("experiment_config"):
                    reasons.append("paired_experiment_config_mismatch")
                if (
                    none_row.get("block_id") != burst_row.get("block_id")
                    or none_row.get("protocol") != burst_row.get("protocol")
                    or none_row.get("seed") != burst_row.get("seed")
                ):
                    reasons.append("paired_identity_mismatch")
                if not reasons:
                    valid_pairs.append({
                        "seed": seed,
                        "active_text_lanes": lane_count,
                        "none": none_row,
                        "burst": burst_row,
                    })
            pair_audit.append({
                "seed": seed,
                "active_text_lanes": lane_count,
                "attempts_none": len(mode_attempts["none"]),
                "eligible_none": len(none_eligible),
                "attempts_burst": len(mode_attempts["burst"]),
                "eligible_burst": len(burst_eligible),
                "ineligible_attempts_present": (
                    len(mode_attempts["none"]) != len(none_eligible)
                    or len(mode_attempts["burst"]) != len(burst_eligible)
                ),
                "paired": not reasons,
                "reasons": reasons,
                "selected_none_run_id": none_eligible[0].get("run_id") if len(none_eligible) == 1 else None,
                "selected_burst_run_id": burst_eligible[0].get("run_id") if len(burst_eligible) == 1 else None,
            })

    def summarize_contrasts(
        pairs: list[dict[str, Any]], paths: dict[str, tuple[str, ...]], lane_count: int, salt: int
    ) -> tuple[dict[str, Any], dict[str, dict[int, float]]]:
        summaries_by_metric: dict[str, Any] = {}
        differences_by_metric: dict[str, dict[int, float]] = {}
        for metric_name, metric_path in paths.items():
            paired_values: list[tuple[int, float]] = []
            for pair in pairs:
                if pair["active_text_lanes"] != lane_count:
                    continue
                none_value = at_path(pair["none"], metric_path)
                burst_value = at_path(pair["burst"], metric_path)
                if none_value is not None and burst_value is not None:
                    paired_values.append((pair["seed"], burst_value - none_value))
            diffs = [difference for _, difference in paired_values]
            differences_by_metric[metric_name] = dict(paired_values)
            ci = None
            if len(diffs) >= 2:
                rng = random.Random(salt + lane_count)
                bootstrap_means = [
                    statistics.mean(rng.choices(diffs, k=len(diffs)))
                    for _ in range(10_000)
                ]
                ci = [
                    quantile_nearest_rank(bootstrap_means, 0.025),
                    quantile_nearest_rank(bootstrap_means, 0.975),
                ]
            summaries_by_metric[metric_name] = {
                "estimand": "burst minus none; paired by seed, run is the replicate",
                "paired_runs": len(diffs),
                "by_seed": {str(seed): difference for seed, difference in paired_values},
                "mean_difference": statistics.mean(diffs) if diffs else None,
                "median_difference": statistics.median(diffs) if diffs else None,
                "bootstrap_95pct_ci_mean": ci,
                "bootstrap_note": "not estimated for fewer than two paired seeds" if len(diffs) < 2 else None,
            }
        return summaries_by_metric, differences_by_metric

    paired_effects: dict[str, Any] = {}
    supportive_anchor_contrasts: dict[str, Any] = {}
    effect_differences_by_lane: dict[int, dict[str, dict[int, float]]] = {}
    anchor_metric_path = {
        "anchor_window_active_request_chunk_gap_p95_ms": (
            "text", "anchor_local_time_association", "request_internal_gaps_overlapping_windows_ms", "p95"
        )
    }
    anchor_differences_by_lane: dict[int, dict[str, dict[int, float]]] = {}
    for lane_count in (2, 6):
        paired_effects[str(lane_count)], effect_differences_by_lane[lane_count] = summarize_contrasts(
            valid_pairs, metric_paths, lane_count, 20261111
        )
        supportive_anchor_contrasts[str(lane_count)], anchor_differences_by_lane[lane_count] = summarize_contrasts(
            valid_pairs, anchor_metric_path, lane_count, 20261121
        )

    pair_by_seed_lane = {
        (int(pair["seed"]), int(pair["active_text_lanes"])): pair
        for pair in valid_pairs
    }
    interaction_seed_audit = []
    valid_interaction_seeds: set[int] = set()
    for seed in sorted({pair["seed"] for pair in valid_pairs}):
        pair_2 = pair_by_seed_lane.get((seed, 2))
        pair_6 = pair_by_seed_lane.get((seed, 6))
        if pair_2 is None or pair_6 is None:
            continue
        rows_for_seed = [pair_2["none"], pair_2["burst"], pair_6["none"], pair_6["burst"]]
        fingerprints = [row.get("artifact_fingerprints") for row in rows_for_seed]
        configs = [row.get("experiment_config") for row in rows_for_seed]
        reasons = []
        if any(value != fingerprints[0] for value in fingerprints[1:]):
            reasons.append("cross_lane_artifact_fingerprint_mismatch")
        if any(value != configs[0] for value in configs[1:]):
            reasons.append("cross_lane_experiment_config_mismatch")
        if not reasons:
            valid_interaction_seeds.add(seed)
        interaction_seed_audit.append({
            "seed": seed,
            "eligible_for_lane_interaction": not reasons,
            "reasons": reasons,
            "run_ids": [row.get("run_id") for row in rows_for_seed],
        })

    lane_interaction: dict[str, Any] = {}
    for metric_name in metric_paths:
        two_lane = effect_differences_by_lane.get(2, {}).get(metric_name, {})
        six_lane = effect_differences_by_lane.get(6, {}).get(metric_name, {})
        candidate_seeds = sorted(set(two_lane) & set(six_lane))
        shared_seeds = [seed for seed in candidate_seeds if seed in valid_interaction_seeds]
        interaction_values = [six_lane[seed] - two_lane[seed] for seed in shared_seeds]
        if interaction_values:
            rng = random.Random(20261117)
            bootstrap_means = (
                [statistics.mean(rng.choices(interaction_values, k=len(interaction_values))) for _ in range(10_000)]
                if len(interaction_values) >= 2 else []
            )
            lane_interaction[metric_name] = {
                "estimand": "(burst-none at 6 lanes) minus (burst-none at 2 lanes), paired by shared seed",
                "candidate_shared_seeds": [str(seed) for seed in candidate_seeds],
                "paired_seeds": [str(seed) for seed in shared_seeds],
                "by_seed": {
                    str(seed): six_lane[seed] - two_lane[seed] for seed in shared_seeds
                },
                "mean_difference_in_differences": statistics.mean(interaction_values),
                "bootstrap_95pct_ci_mean": (
                    [quantile_nearest_rank(bootstrap_means, 0.025), quantile_nearest_rank(bootstrap_means, 0.975)]
                    if bootstrap_means else None
                ),
                "bootstrap_note": "not estimated for fewer than two paired seeds" if len(interaction_values) < 2 else None,
            }
        else:
            lane_interaction[metric_name] = {
                "estimand": "(burst-none at 6 lanes) minus (burst-none at 2 lanes), paired by shared seed",
                "candidate_shared_seeds": [str(seed) for seed in candidate_seeds],
                "paired_seeds": [],
                "by_seed": {},
                "mean_difference_in_differences": None,
                "bootstrap_95pct_ci_mean": None,
            }
    output = {
        "schema_version": 1,
        "metric_version": "motivation-phase1-v0.1-discovery",
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "measurement_window": "[measurement_start_ns, measurement_start_ns + measurement_seconds)",
        "text_gap_rule": "maximum over adjacent text token timestamps for which both timestamps lie in the exact measurement window; censored boundary intervals are excluded",
        "runs": summaries,
        "phase1_attempt_audit": attempt_audit,
        "phase1_pair_audit": pair_audit,
        "phase1_paired_run_level_outcomes": paired_effects,
        "phase1_supportive_anchor_local_contrasts": supportive_anchor_contrasts,
        "phase1_lane_interaction_seed_audit": interaction_seed_audit,
        "phase1_lane_interaction_difference_in_differences": lane_interaction,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summarized_runs": len(summaries)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
