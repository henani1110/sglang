#!/usr/bin/env python3
"""Admission-side text/image deadline controller for the E request API.

This module observes text stream token timestamps and prepares one image
request's mm_processor_kwargs before the server expands image placeholders.
It cannot enable pruning until content, K_safe, and cost calibration are
explicitly supplied.
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from e_content_budget_policy import (
    PreencoderBudgetConfig,
    choose_joint_preencoder_keep_count,
    query_complexity_proxy,
    request_ratio_for_exact_keep_count,
    route_preencoder_content_budget,
)


class TextDeadlineTracker:
    """Thread-safe last-token timestamps for active streamed text lanes."""

    def __init__(self) -> None:
        self._last_token_ns: dict[str, int] = {}
        self._lock = threading.Lock()

    def record_token(self, lane_id: str, now_ns: int | None = None) -> None:
        if not lane_id:
            raise ValueError("lane_id must be non-empty")
        timestamp = time.monotonic_ns() if now_ns is None else int(now_ns)
        with self._lock:
            self._last_token_ns[lane_id] = timestamp

    def finish_lane(self, lane_id: str) -> None:
        with self._lock:
            self._last_token_ns.pop(lane_id, None)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._last_token_ns)

    def tightest_slack_ms(self, now_ns: int, max_gap_ms: float) -> float | None:
        if not math.isfinite(float(max_gap_ms)) or max_gap_ms <= 0:
            raise ValueError("max_gap_ms must be finite and positive")
        last_tokens = self.snapshot()
        if not last_tokens:
            return None
        return min(
            float(max_gap_ms) - (int(now_ns) - last_token_ns) / 1_000_000.0
            for last_token_ns in last_tokens.values()
        )


def prepare_image_request_budget(
    *,
    query: str,
    pixel_density_score: float | None,
    full_token_count: int,
    image_arrival_ns: int,
    tracker: TextDeadlineTracker,
    now_ns: int | None = None,
    text_max_gap_ms: float = 30.0,
    image_ttft_ms: float = 55.0,
    safe_floor_count: int | None = None,
    candidate_keep_counts: Sequence[int] | None = None,
    projected_text_impact_ms: Mapping[int, float] | None = None,
    projected_image_ttft_ms: Mapping[int, float] | None = None,
    cost_curve_id: str | None = None,
    policy_config: PreencoderBudgetConfig = PreencoderBudgetConfig(),
) -> dict[str, Any]:
    """Build a request-scoped ratio using live D/I slack and fail-safe defaults."""
    if full_token_count < 1:
        raise ValueError("full_token_count must be positive")
    if not math.isfinite(float(text_max_gap_ms)) or text_max_gap_ms <= 0:
        raise ValueError("text_max_gap_ms must be finite and positive")
    if not math.isfinite(float(image_ttft_ms)) or image_ttft_ms <= 0:
        raise ValueError("image_ttft_ms must be finite and positive")
    now = time.monotonic_ns() if now_ns is None else int(now_ns)
    arrival = int(image_arrival_ns)
    if arrival > now:
        raise ValueError("image_arrival_ns cannot be after now_ns")

    query_features = query_complexity_proxy(query)
    route = route_preencoder_content_budget(
        query_simple_proxy=query_features["query_simple_proxy"],
        pixel_density_score=pixel_density_score,
        config=policy_config,
    )

    raw_text_slack = tracker.tightest_slack_ms(now, text_max_gap_ms)
    text_lanes = tracker.snapshot()
    raw_image_slack = float(image_ttft_ms) - (now - arrival) / 1_000_000.0
    image_overdue_ms = max(0.0, -raw_image_slack)
    image_slack = max(0.0, raw_image_slack)

    curves_ready = (
        safe_floor_count is not None
        and candidate_keep_counts is not None
        and projected_text_impact_ms is not None
        and projected_image_ttft_ms is not None
        and bool(cost_curve_id)
    )
    calibration_ready = policy_config.calibrated_for_current_backbone and curves_ready
    route_can_prune = route["route"] in {"aggressive_proxy", "conservative_proxy"}

    joint_decision: dict[str, Any] | None = None
    if calibration_ready and route_can_prune:
        assert safe_floor_count is not None
        assert candidate_keep_counts is not None
        assert projected_text_impact_ms is not None
        assert projected_image_ttft_ms is not None
        if not 1 <= safe_floor_count <= full_token_count:
            raise ValueError("safe_floor_count must be within [1, full_token_count]")
        if raw_text_slack is None:
            # With no active text stream, remove the text constraint while
            # keeping a finite, auditable value in the decision record.
            effective_text_slack = max(
                (float(v) for v in projected_text_impact_ms.values()), default=0.0
            )
            text_constraint_active = False
            text_overdue_ms = 0.0
        else:
            effective_text_slack = max(0.0, raw_text_slack)
            text_constraint_active = True
            text_overdue_ms = max(0.0, -raw_text_slack)
        joint_decision = choose_joint_preencoder_keep_count(
            full_token_count=full_token_count,
            safe_floor_count=safe_floor_count,
            candidate_keep_counts=candidate_keep_counts,
            content_keep_ratio_cap=float(route["effective_keep_ratio"]),
            projected_text_impact_ms=projected_text_impact_ms,
            projected_image_ttft_ms=projected_image_ttft_ms,
            text_slack_ms=effective_text_slack,
            image_slack_ms=image_slack,
        )
        joint_decision["text_constraint_active"] = text_constraint_active
        joint_decision["text_slack_raw_ms"] = raw_text_slack
        joint_decision["text_overdue_ms"] = text_overdue_ms
        joint_decision["image_slack_raw_ms"] = raw_image_slack
        joint_decision["image_overdue_ms"] = image_overdue_ms
        joint_decision["cost_curve_id"] = cost_curve_id
        chosen_count = int(joint_decision["chosen_keep_count"])
        decision_reason = "calibrated_content_cap_and_live_deadline_slack"
    else:
        # Dense bypass, missing signals, missing calibration, or missing cost
        # curves never permit deadline pressure to override the keep-all guard.
        chosen_count = full_token_count
        decision_reason = (
            "content_dense_or_signal_missing_keep_all"
            if route["route"] in {"dense_pixel_bypass", "insufficient_signals"}
            else "missing_current_backbone_quality_or_cost_calibration_keep_all"
        )
        text_overdue_ms = max(0.0, -(raw_text_slack or 0.0))

    ratio = request_ratio_for_exact_keep_count(full_token_count, chosen_count)
    return {
        "request_budget_active": bool(calibration_ready and route_can_prune),
        "decision_reason": decision_reason,
        "query_features": query_features,
        "content_route": route,
        "text_lane_count": len(text_lanes),
        "text_slack_raw_ms": raw_text_slack,
        "text_overdue_ms": text_overdue_ms,
        "image_slack_raw_ms": raw_image_slack,
        "image_overdue_ms": image_overdue_ms,
        "joint_decision": joint_decision,
        "full_token_count": full_token_count,
        "chosen_keep_count": chosen_count,
        "request_keep_ratio": ratio,
        "mm_processor_kwargs": {"images_kwargs": {"e_keep_ratio": ratio}},
    }


def self_check() -> dict[str, Any]:
    now = 10_000_000_000
    tracker = TextDeadlineTracker()
    tracker.record_token("text-a", now - 5_000_000)
    tracker.record_token("text-b", now - 12_000_000)
    assert tracker.tightest_slack_ms(now, 30.0) == 18.0

    cfg_uncalibrated = PreencoderBudgetConfig()
    fallback = prepare_image_request_budget(
        query="What animal is shown?",
        pixel_density_score=0.01,
        full_token_count=256,
        image_arrival_ns=now,
        now_ns=now,
        tracker=tracker,
        safe_floor_count=128,
        candidate_keep_counts=[128, 160, 192, 256],
        projected_text_impact_ms={128: 1.0, 160: 2.0, 192: 4.0, 256: 8.0},
        projected_image_ttft_ms={128: 1.5, 160: 3.0, 192: 6.0, 256: 9.0},
        cost_curve_id="synthetic-only",
        policy_config=cfg_uncalibrated,
    )
    assert fallback["chosen_keep_count"] == 256
    assert fallback["request_keep_ratio"] == 1.0
    assert fallback["text_slack_raw_ms"] == 18.0
    assert not fallback["request_budget_active"]

    cfg_calibrated = PreencoderBudgetConfig(
        query_simple_threshold=0.70,
        pixel_dense_threshold=0.03,
        aggressive_keep_ratio=0.75,
        conservative_keep_ratio=0.85,
        calibrated_for_current_backbone=True,
        calibration_id="synthetic-branch-check-only",
    )
    tight_tracker = TextDeadlineTracker()
    tight_tracker.record_token("text-lane", now - 26_500_000)
    active = prepare_image_request_budget(
        query="What animal is shown?",
        pixel_density_score=0.01,
        full_token_count=256,
        image_arrival_ns=now - 48_000_000,
        now_ns=now,
        tracker=tight_tracker,
        safe_floor_count=128,
        candidate_keep_counts=[64, 128, 160, 192, 256],
        projected_text_impact_ms={128: 1.0, 160: 2.0, 192: 4.0, 256: 8.0},
        projected_image_ttft_ms={128: 1.5, 160: 3.0, 192: 6.0, 256: 9.0},
        cost_curve_id="synthetic-only",
        policy_config=cfg_calibrated,
    )
    assert active["request_budget_active"]
    assert active["text_slack_raw_ms"] == 3.5
    assert active["image_slack_raw_ms"] == 7.0
    assert active["chosen_keep_count"] == 160
    assert math.ceil(256 * active["request_keep_ratio"]) == 160

    dense = prepare_image_request_budget(
        query="What animal is shown?",
        pixel_density_score=0.10,
        full_token_count=256,
        image_arrival_ns=now,
        now_ns=now,
        tracker=tight_tracker,
        safe_floor_count=128,
        candidate_keep_counts=[128, 160, 192, 256],
        projected_text_impact_ms={128: 1.0, 160: 2.0, 192: 4.0, 256: 8.0},
        projected_image_ttft_ms={128: 1.5, 160: 3.0, 192: 6.0, 256: 9.0},
        cost_curve_id="synthetic-only",
        policy_config=cfg_calibrated,
    )
    assert dense["chosen_keep_count"] == 256
    assert not dense["request_budget_active"]

    overdue = prepare_image_request_budget(
        query="What animal is shown?",
        pixel_density_score=0.01,
        full_token_count=256,
        image_arrival_ns=now - 60_000_000,
        now_ns=now,
        tracker=tight_tracker,
        safe_floor_count=128,
        candidate_keep_counts=[128, 160, 192, 256],
        projected_text_impact_ms={128: 1.0, 160: 2.0, 192: 4.0, 256: 8.0},
        projected_image_ttft_ms={128: 1.5, 160: 3.0, 192: 6.0, 256: 9.0},
        cost_curve_id="synthetic-only",
        policy_config=cfg_calibrated,
    )
    assert overdue["chosen_keep_count"] == 128
    assert overdue["image_overdue_ms"] == 5.0
    tracker.finish_lane("text-a")
    tracker.finish_lane("text-b")
    assert tracker.tightest_slack_ms(now, 30.0) is None
    return {
        "valid": True,
        "checks": {
            "multi_lane_tightest_text_slack": True,
            "uncalibrated_request_is_keep_all": True,
            "calibrated_joint_choice_obeys_both_deadlines_and_content_cap": True,
            "dense_proxy_vetoes_pruning": True,
            "missed_deadline_preserves_quality_floor_and_reports_overdue": True,
            "finished_lanes_leave_snapshot": True,
        },
        "limitations": [
            "Synthetic costs and thresholds only exercise branches; no calibration or answer-quality claim.",
            "The module is an admission-side controller primitive and is not connected to the formal load generator yet.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if not args.self_check:
        parser.error("select --self-check; serving traffic is not enabled by this module")
    report = self_check()
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
