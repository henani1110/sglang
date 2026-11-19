#!/usr/bin/env python3
"""Summarize the fixed vLLM visual-encoding pilot logs."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    fraction = position - left
    return ordered[left] * (1 - fraction) + ordered[right] * fraction


def request_keys(mapping: dict[str, Any], request_id: str) -> list[str]:
    return [
        key
        for key in mapping
        if key == request_id or key.startswith(request_id + "-")
    ]


def gpu_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"rows": 0, "gpus": {}}
    samples: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            clean = {key.strip(): value.strip() for key, value in row.items() if key}
            if clean.get("gpu_index"):
                samples[clean["gpu_index"]].append(clean)
    out: dict[str, Any] = {"rows": sum(map(len, samples.values())), "gpus": {}}
    for index, rows in samples.items():
        def numbers(field: str) -> list[float]:
            result = []
            for row in rows:
                try:
                    result.append(float(row[field]))
                except (KeyError, TypeError, ValueError):
                    pass
            return result

        out["gpus"][index] = {
            "uuid": rows[0].get("uuid"),
            "samples": len(rows),
            "max_utilization_gpu_percent": max(numbers("utilization_gpu_percent"), default=None),
            "max_memory_used_mib": max(numbers("memory_used_mib"), default=None),
            "max_temperature_c": max(numbers("temperature_c"), default=None),
            "max_power_w": max(numbers("power_w"), default=None),
            "pci_link_speeds": sorted({row.get("pci_link_speed") for row in rows}),
            "pci_link_widths": sorted({row.get("pci_link_width") for row in rows}),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--client-events", type=Path, required=True)
    parser.add_argument("--server-events", type=Path, required=True)
    parser.add_argument("--scheduler-events", type=Path, required=True)
    parser.add_argument("--gpu-monitor", type=Path, required=True)
    parser.add_argument("--block-count", type=int, default=6)
    parser.add_argument("--device-label", default="GPU0")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.results.read_text())
    client_events = read_jsonl(args.client_events)
    server_events = read_jsonl(args.server_events)
    scheduler_events = read_jsonl(args.scheduler_events)

    server_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in server_events:
        if event.get("kind") == "server_token_ready" and event.get("token_ids"):
            server_by_request[str(event.get("request_id"))].append(event)
    for events in server_by_request.values():
        events.sort(key=lambda item: int(item["time_ns"]))

    all_episodes = result.get("episodes", [])
    run_start = min((int(item["start_ns"]) for item in all_episodes if item.get("start_ns")), default=0)
    run_end = max((int(item["end_ns"]) for item in all_episodes if item.get("end_ns")), default=0)
    visual_events = [
        event
        for event in server_events
        if event.get("kind") == "visual_forward"
        and run_start <= int(event.get("time_ns", 0)) <= run_end
    ]
    client_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in client_events:
        if event.get("episode_id"):
            client_by_episode[str(event["episode_id"])].append(event)

    measured = [episode for episode in result.get("episodes", []) if episode.get("arm") in {"T", "R", "B"}]
    episode_metrics: list[dict[str, Any]] = []
    gaps_by_arm: dict[str, list[float]] = defaultdict(list)
    prewindow_gaps_by_arm: dict[str, list[float]] = defaultdict(list)
    client_gaps_by_arm: dict[str, list[float]] = defaultdict(list)

    for episode in measured:
        anchor = int(episode["anchor_ns"])
        low = anchor - 50_000_000
        high = anchor + 400_000_000
        lane_data = []
        full_gaps = []
        prewindow_gaps = []
        window_gaps = []
        coverage = []
        client_coverage = []
        client_window_gaps = []
        text_items = episode.get("text", [])
        for lane, item in enumerate(text_items):
            request_id = str(item.get("request_id") or "")
            events = server_by_request.get(request_id, [])
            times = [int(event["time_ns"]) for event in events]
            intervals = [
                (times[index - 1], times[index], (times[index] - times[index - 1]) / 1e6)
                for index in range(1, len(times))
            ]
            full_gaps.extend(gap for _, _, gap in intervals)
            before = [gap for left, right, gap in intervals if right <= low]
            in_window = [gap for left, right, gap in intervals if left <= high and right >= low]
            prewindow_gaps.extend(before)
            window_gaps.extend(in_window)
            lane_coverage = bool(any(time_ns <= low for time_ns in times) and any(time_ns >= high for time_ns in times))
            coverage.append(lane_coverage)
            client_lane_events = [
                event
                for event in client_by_episode.get(str(episode["episode_id"]), [])
                if event.get("kind") == "sse_chunk"
                and event.get("lane") == lane
                and event.get("token_ids")
                and event.get("request_id") == request_id
            ]
            client_times = sorted(int(event["time_ns"]) for event in client_lane_events)
            client_coverage.append(
                bool(any(time_ns <= low for time_ns in client_times) and any(time_ns >= high for time_ns in client_times))
            )
            client_intervals = [
                (client_times[index] - client_times[index - 1]) / 1e6
                for index in range(1, len(client_times))
            ]
            client_gaps_by_arm[episode["arm"]].extend(client_intervals)
            client_window_gaps.extend(
                (client_times[index] - client_times[index - 1]) / 1e6
                for index in range(1, len(client_times))
                if client_times[index - 1] <= high and client_times[index] >= low
            )
            lane_data.append(
                {
                    "lane": lane,
                    "request_id": request_id,
                    "ready_events": len(events),
                    "tokens_before_window": sum(
                        len(event.get("token_ids", [])) for event in events if int(event["time_ns"]) <= low
                    ),
                    "tokens_through_window": sum(
                        len(event.get("token_ids", [])) for event in events if int(event["time_ns"]) <= high
                    ),
                    "window_covered": lane_coverage,
                    "client_window_covered": client_coverage[-1],
                }
            )
        gaps_by_arm[episode["arm"]].extend(full_gaps)
        prewindow_gaps_by_arm[episode["arm"]].extend(prewindow_gaps)
        episode_metrics.append(
            {
                "block": episode["block"],
                "arm": episode["arm"],
                "trial": episode["trial"],
                "episode_id": episode["episode_id"],
                "anchor_ns": anchor,
                "arrival_offset_ms": episode.get("arrival_offset_ms"),
                "trigger_counts": episode.get("text_token_counts_at_trigger"),
                "anchor_counts": episode.get("text_token_counts_at_anchor"),
                "active_at_anchor": episode.get("active_text_streams_at_anchor"),
                "all_four_lanes_cover_window": len(coverage) == 4 and all(coverage),
                "all_four_client_lanes_cover_window": len(client_coverage) == 4 and all(client_coverage),
                "covered_lane_count": sum(coverage),
                "window_gap_count": len(window_gaps),
                "max_window_gap_ms": max(window_gaps, default=None),
                "p95_window_gap_ms": quantile(window_gaps, 0.95),
                "max_prewindow_gap_ms": max(prewindow_gaps, default=None),
                "p95_prewindow_gap_ms": quantile(prewindow_gaps, 0.95),
                "client_window_gap_count": len(client_window_gaps),
                "client_max_window_gap_ms": max(client_window_gaps, default=None),
                "full_stream_gap_count": len(full_gaps),
                "prewindow_gap_count": len(prewindow_gaps),
                "lanes": lane_data,
                "image": episode.get("image"),
            }
        )

    by_block_arm: dict[str, dict[str, Any]] = {}
    for block in range(1, args.block_count + 1):
        by_block_arm[str(block)] = {}
        for arm in ("T", "R", "B"):
            rows = [item for item in episode_metrics if item["block"] == block and item["arm"] == arm]
            values = [float(item["max_window_gap_ms"]) for item in rows if item["max_window_gap_ms"] is not None]
            pre_values = [
                float(item["max_prewindow_gap_ms"])
                for item in rows
                if item["max_prewindow_gap_ms"] is not None
            ]
            by_block_arm[str(block)][arm] = {
                "episodes": len(rows),
                "window_coverage_rate": sum(item["all_four_lanes_cover_window"] for item in rows) / len(rows) if rows else None,
                "window_max_gap_p95_ms": quantile(values, 0.95),
                "window_max_gap_median_ms": statistics.median(values) if values else None,
                "window_max_gap_range_ms": [min(values), max(values)] if values else None,
                "prewindow_max_gap_p95_ms": quantile(pre_values, 0.95),
            }

    paired = []
    for block, values in by_block_arm.items():
        r_value = values["R"]["window_max_gap_p95_ms"]
        b_value = values["B"]["window_max_gap_p95_ms"]
        t_value = values["T"]["window_max_gap_p95_ms"]
        r_prewindow = values["R"]["prewindow_max_gap_p95_ms"]
        b_prewindow = values["B"]["prewindow_max_gap_p95_ms"]
        paired.append(
            {
                "block": int(block),
                "R_p95_ms": r_value,
                "B_p95_ms": b_value,
                "T_p95_ms": t_value,
                "R_minus_B_ms": r_value - b_value if r_value is not None and b_value is not None else None,
                "R_vs_B_percent": (r_value - b_value) / b_value * 100 if r_value is not None and b_value else None,
                "B_minus_T_ms": b_value - t_value if b_value is not None and t_value is not None else None,
                "R_minus_B_prewindow_ms": r_prewindow - b_prewindow if r_prewindow is not None and b_prewindow is not None else None,
                "R_prewindow_p95_ms": r_prewindow,
                "B_prewindow_p95_ms": b_prewindow,
            }
        )
    differences = [row["R_minus_B_ms"] for row in paired if row["R_minus_B_ms"] is not None]
    relative = [row["R_vs_B_percent"] for row in paired if row["R_vs_B_percent"] is not None]
    prewindow_differences = [
        row["R_minus_B_prewindow_ms"]
        for row in paired
        if row["R_minus_B_prewindow_ms"] is not None
    ]

    request_schedule: dict[str, dict[str, Any]] = {}
    for episode in measured:
        if episode.get("arm") == "T" or not episode.get("image"):
            continue
        image_id = str(episode["image"].get("request_id") or "")
        text_ids = [str(item.get("request_id") or "") for item in episode.get("text", [])]
        first = None
        for event in scheduler_events:
            enc = event.get("scheduled_encoder_inputs", {})
            if request_keys(enc, image_id) and int(event.get("time_ns", 0)) >= int(episode["anchor_ns"]) - 100_000_000:
                first = event
                break
        if first is None:
            request_schedule[episode["episode_id"]] = {"found_first_image_schedule": False}
            continue
        scheduled = first.get("num_scheduled_tokens", {})
        image_keys = request_keys(scheduled, image_id)
        encoder_keys = request_keys(first.get("scheduled_encoder_inputs", {}), image_id)
        lane_counts = []
        lane_present = []
        for request_id in text_ids:
            keys = request_keys(scheduled, request_id)
            lane_present.append(bool(keys))
            lane_counts.append(sum(int(scheduled[key]) for key in keys))
        ready_before = []
        for request_id in text_ids:
            ready_before.append(
                sum(
                    len(row.get("token_ids", []))
                    for row in server_by_request.get(request_id, [])
                    if int(row["time_ns"]) < int(first["time_ns"])
                )
            )
        request_schedule[episode["episode_id"]] = {
            "found_first_image_schedule": True,
            "time_ns": first["time_ns"],
            "offset_from_anchor_ms": (int(first["time_ns"]) - int(episode["anchor_ns"])) / 1e6,
            "image_request_keys": request_keys(first.get("scheduled_encoder_inputs", {}), image_id),
            "image_scheduled_tokens": sum(int(scheduled[key]) for key in image_keys),
            "image_encoder_input_indices": [
                int(index)
                for key in encoder_keys
                for index in first.get("scheduled_encoder_inputs", {}).get(key, [])
            ],
            "all_four_text_requests_present": all(lane_present),
            "text_scheduled_tokens_by_lane": lane_counts,
            "total_scheduled_tokens": first.get("total_num_scheduled_tokens"),
            "text_tokens_ready_before_mixed_step_by_lane": ready_before,
        }

    pair_matching = []
    episodes_by_key = {(episode["block"], episode["trial"], episode["arm"]): episode for episode in measured}
    output_equivalence = []
    for block in range(1, args.block_count + 1):
        for trial in range(16):
            r_episode = episodes_by_key.get((block, trial, "R"))
            b_episode = episodes_by_key.get((block, trial, "B"))
            if not r_episode or not b_episode:
                continue
            r_text = r_episode.get("text", [])
            b_text = b_episode.get("text", [])
            text_lane_matches = [
                left.get("token_ids") == right.get("token_ids")
                and left.get("prompt_token_ids") == right.get("prompt_token_ids")
                for left, right in zip(r_text, b_text)
            ]
            r_image = r_episode.get("image") or {}
            b_image = b_episode.get("image") or {}
            output_equivalence.append(
                {
                    "block": block,
                    "trial": trial,
                    "all_four_text_outputs_equal": len(text_lane_matches) == 4 and all(text_lane_matches),
                    "image_outputs_equal": r_image.get("token_ids") == b_image.get("token_ids"),
                    "text_lane_matches": text_lane_matches,
                }
            )
            r_sched = request_schedule.get(r_episode["episode_id"], {})
            b_sched = request_schedule.get(b_episode["episode_id"], {})
            r_ready = r_sched.get("text_tokens_ready_before_mixed_step_by_lane")
            b_ready = b_sched.get("text_tokens_ready_before_mixed_step_by_lane")
            ready_diffs = [abs(a - b) for a, b in zip(r_ready or [], b_ready or [])]
            max_ready_diff = max(ready_diffs, default=None)
            signature_match = (
                r_sched.get("text_scheduled_tokens_by_lane") == b_sched.get("text_scheduled_tokens_by_lane")
                and r_sched.get("image_scheduled_tokens") == b_sched.get("image_scheduled_tokens")
                and r_sched.get("image_encoder_input_indices") == b_sched.get("image_encoder_input_indices")
                and r_sched.get("total_scheduled_tokens") == b_sched.get("total_scheduled_tokens")
            )
            matched = bool(
                r_sched.get("found_first_image_schedule")
                and b_sched.get("found_first_image_schedule")
                and r_sched.get("all_four_text_requests_present")
                and b_sched.get("all_four_text_requests_present")
                and signature_match
                and max_ready_diff is not None
                and max_ready_diff <= 2
            )
            pair_matching.append(
                {
                    "block": block,
                    "trial": trial,
                    "matched": matched,
                    "scheduler_signature_match": signature_match,
                    "max_text_tokens_ready_difference": max_ready_diff,
                    "R": r_sched,
                    "B": b_sched,
                }
            )

    arm_summaries = {}
    for arm in ("T", "R", "B"):
        values = gaps_by_arm[arm]
        pre_values = prewindow_gaps_by_arm[arm]
        rows = [item for item in episode_metrics if item["arm"] == arm]
        arm_summaries[arm] = {
            "episodes": len(rows),
            "full_stream_gap_count": len(values),
            "full_stream_interval_p95_ms": quantile(values, 0.95),
            "full_stream_interval_p99_ms": quantile(values, 0.99),
            "prewindow_interval_p95_ms": quantile(pre_values, 0.95),
            "all_four_lane_window_coverage_rate": sum(item["all_four_lanes_cover_window"] for item in rows) / len(rows) if rows else None,
            "all_four_client_lane_window_coverage_rate": sum(item["all_four_client_lanes_cover_window"] for item in rows) / len(rows) if rows else None,
            "client_sse_interval_p95_ms": quantile(client_gaps_by_arm[arm], 0.95),
            "client_sse_interval_p99_ms": quantile(client_gaps_by_arm[arm], 0.99),
            "validations_failed": sum(bool(item.get("validation_errors")) for item in measured if item["arm"] == arm),
        }

    feature_validation = {
        "visual_events": len(visual_events),
        "by_mode": {
            mode: sum(event.get("mode") == mode for event in visual_events)
            for mode in ("NORMAL", "CAPTURE", "VERIFY", "R", "B")
        },
        "formal_verify_events": sum(event.get("mode") == "VERIFY" for event in visual_events),
        "formal_verify_bitwise_passes": sum(
            event.get("mode") == "VERIFY"
            and event.get("check", {}).get("pass") is True
            and event.get("check", {}).get("bitwise_equal") is True
            for event in visual_events
        ),
        "B_resident_feature_hits": sum(
            event.get("mode") == "B"
            and event.get("check", {}).get("kind") == "resident_feature"
            and event.get("check", {}).get("pass") is True
            for event in visual_events
        ),
    }

    report = {
        "device": args.device_label,
        "block_count": args.block_count,
        "source_status": result.get("status"),
        "model": result.get("model"),
        "text_prompt_token_lengths": result.get("text_prompt_token_lengths"),
        "text_output_tokens": result.get("text_output_tokens"),
        "image_output_tokens": result.get("image_output_tokens"),
        "episode_count_total": len(result.get("episodes", [])),
        "measured_episode_count": len(measured),
        "episodes_by_block_arm": by_block_arm,
        "paired_block_differences": paired,
        "R_minus_B_median_ms": statistics.median(differences) if differences else None,
        "R_minus_B_median_percent": statistics.median(relative) if relative else None,
        "R_minus_B_positive_blocks": sum(value > 0 for value in differences),
        "R_minus_B_prewindow_median_ms": statistics.median(prewindow_differences) if prewindow_differences else None,
        "arm_summaries": arm_summaries,
        "episode_metrics": episode_metrics,
        "image_mixed_step_by_episode": request_schedule,
        "R_B_pair_matching": {
            "pairs": len(pair_matching),
            "matched_pairs": sum(item["matched"] for item in pair_matching),
            "matched_rate": sum(item["matched"] for item in pair_matching) / len(pair_matching) if pair_matching else None,
            "pairs_with_same_first_step_signature": sum(item["scheduler_signature_match"] for item in pair_matching),
            "details": pair_matching,
        },
        "R_B_output_equivalence": {
            "pairs": len(output_equivalence),
            "all_text_outputs_equal_pairs": sum(item["all_four_text_outputs_equal"] for item in output_equivalence),
            "image_outputs_equal_pairs": sum(item["image_outputs_equal"] for item in output_equivalence),
            "details": output_equivalence,
        },
        "feature_validation": feature_validation,
        "gpu_monitor": gpu_summary(args.gpu_monitor),
        "log_counts": {
            "client_events": len(client_events),
            "server_events": len(server_events),
            "scheduler_events": len(scheduler_events),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Visual encoding interference pilot",
        "",
        f"- Device: `{report['device']}`",
        f"- Result status: `{report['source_status']}`",
        f"- Measured episodes: {report['measured_episode_count']} (warmups excluded)",
        f"- R−B positive paired blocks: {report['R_minus_B_positive_blocks']}/{report['block_count']}",
        f"- Median paired R−B P95: {report['R_minus_B_median_ms']} ms",
        f"- Median paired relative change vs B: {report['R_minus_B_median_percent']}%",
        f"- Median paired pre-window R−B P95: {report['R_minus_B_prewindow_median_ms']} ms",
        f"- R/B scheduler matched pairs: {report['R_B_pair_matching']['matched_pairs']}/{report['R_B_pair_matching']['pairs']}",
        f"- R/B mixed output equality: four text streams {report['R_B_output_equivalence']['all_text_outputs_equal_pairs']}/{report['R_B_output_equivalence']['pairs']}; image output {report['R_B_output_equivalence']['image_outputs_equal_pairs']}/{report['R_B_output_equivalence']['pairs']}.",
        "",
        "## Per-block 450 ms window maximum-gap P95",
        "",
        "| Block | T (ms) | R (ms) | B (ms) | R−B (ms) | R−B vs B (%) | Pre-window R−B (ms) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        def fmt(value: Any) -> str:
            return "NA" if value is None else f"{value:.3f}"

        lines.append(
            f"| {row['block']} | {fmt(row['T_p95_ms'])} | {fmt(row['R_p95_ms'])} | "
            f"{fmt(row['B_p95_ms'])} | {fmt(row['R_minus_B_ms'])} | {fmt(row['R_vs_B_percent'])} | "
            f"{fmt(row['R_minus_B_prewindow_ms'])} |"
        )
    lines.extend(
        [
            "",
            "## Coverage and full-stream intervals",
            "",
            "| Arm | Episodes | Server/client four-lane coverage | Server interval P95/P99 (ms) | Client SSE P95/P99 (ms) | Pre-window P95 (ms) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ("T", "R", "B"):
        row = arm_summaries[arm]
        lines.append(
            f"| {arm} | {row['episodes']} | {row['all_four_lane_window_coverage_rate']} / {row['all_four_client_lane_window_coverage_rate']} | "
            f"{row['full_stream_interval_p95_ms']} / {row['full_stream_interval_p99_ms']} | "
            f"{row['client_sse_interval_p95_ms']} / {row['client_sse_interval_p99_ms']} | "
            f"{row['prewindow_interval_p95_ms']} |"
        )
    lines.extend(
        [
            "",
            "## Instrumentation and resource evidence",
            "",
            f"- Formal visual hook events: {feature_validation['visual_events']}; formal R visual computes: {feature_validation['by_mode']['R']}; formal B resident feature hits: {feature_validation['B_resident_feature_hits']}. Feature bitwise verification is in the preflight equivalence artifact.",
            f"- GPU samples: {report['gpu_monitor']['rows']} rows. Per-device utilization, memory, temperature, power and PCIe link states are in the JSON and raw CSV.",
            f"- R/B matching is based on the first scheduler step containing the image encoder input, four active text request IDs, token-count difference ≤2 and exact first-step scheduled-token signature; see JSON for pair-level details.",
            "- This summary does not infer kernel-level overlap from CUDA event duration; the CUDA event/NVTX diagnostics are reported separately.",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
