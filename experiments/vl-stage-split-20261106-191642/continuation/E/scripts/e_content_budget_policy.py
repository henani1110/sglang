#!/usr/bin/env python3
"""Exploratory, fail-safe content-budget and deadline-budget policy primitives.

This module is intentionally not wired into the serving path. Its SieveServe-
inspired routing is inactive until thresholds and keep ratios are calibrated on
the current Qwen3-VL-2B setup and a rights-cleared quality set.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PAPER_REFERENCE = {
    "qcr_model": "ModernBERT-base binary classifier fine-tuned on 9.2K WildVision-Chat queries",
    "qcr_confidence_threshold": 0.80,
    "dig_score": "mean over channels of variance across visual patch embeddings",
    "dig_dense_bypass_rule": "s_DIG < threshold",
    "dig_threshold": 0.55,
    "conservative_drop_ratio": 0.667,
    "aggressive_drop_ratio": 0.778,
    "scope": "SieveServe EMNLP 2026 paper settings; not calibrated for this Qwen3-VL-2B environment",
}


@dataclass(frozen=True)
class ContentPolicyConfig:
    qcr_simple_threshold: float = 0.80
    dig_dense_threshold: float = 0.55
    aggressive_drop_ratio: float = 0.778
    conservative_drop_ratio: float = 0.667
    calibrated_for_current_backbone: bool = False
    calibration_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("qcr_simple_threshold", "dig_dense_threshold", "aggressive_drop_ratio", "conservative_drop_ratio"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.calibrated_for_current_backbone and not self.calibration_id:
            raise ValueError("calibration_id is required when current-backbone calibration is enabled")


@dataclass(frozen=True)
class PreencoderBudgetConfig:
    """Request-side proxy policy; thresholds stay inactive until calibration."""

    query_simple_threshold: float = 0.70
    pixel_dense_threshold: float = 0.03
    aggressive_keep_ratio: float = 0.222
    conservative_keep_ratio: float = 0.333
    calibrated_for_current_backbone: bool = False
    calibration_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "query_simple_threshold",
            "aggressive_keep_ratio",
            "conservative_keep_ratio",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        # With RGB values in [0, 1], a mean per-channel spatial variance is in
        # [0, 0.25]. This is a raw-pixel proxy, not SieveServe's embedding DIG.
        if (
            not math.isfinite(float(self.pixel_dense_threshold))
            or not 0.0 <= float(self.pixel_dense_threshold) <= 0.25
        ):
            raise ValueError("pixel_dense_threshold must be finite and in [0, 0.25]")
        if self.aggressive_keep_ratio > self.conservative_keep_ratio:
            raise ValueError("aggressive keep ratio must not exceed conservative keep ratio")
        if self.calibrated_for_current_backbone and not self.calibration_id:
            raise ValueError("calibration_id is required when current-backbone calibration is enabled")


def raw_pixel_density_proxy(pixels: Any) -> float | list[float]:
    """Mean channel variance across spatial pixels in RGB [0, 1] input.

    Accepts [C,H,W] or [N,C,H,W]. This intentionally runs before E so a
    request's placeholder count is known before scheduler admission. It is not
    an embedding-space DIG score and must be recalibrated after preprocessing.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError("PyTorch is required for pixel density proxy") from exc
    if not isinstance(pixels, torch.Tensor) or pixels.ndim not in (3, 4):
        raise ValueError("pixels must be a torch.Tensor shaped [C,H,W] or [N,C,H,W]")
    batch = pixels.unsqueeze(0) if pixels.ndim == 3 else pixels
    if batch.shape[0] < 1 or batch.shape[1] not in (1, 3) or min(batch.shape[-2:]) < 2:
        raise ValueError("pixels need at least one image, 1/3 channels, and spatial size >= 2")
    values = batch.detach().float()
    if not torch.isfinite(values).all():
        raise ValueError("pixels must be finite")
    if float(values.min().item()) < -1e-6 or float(values.max().item()) > 1.000001:
        raise ValueError("raw_pixel_density_proxy expects RGB values in [0, 1]")
    scores = values.var(dim=(-2, -1), unbiased=False).mean(dim=1)
    result = [float(value) for value in scores.cpu().tolist()]
    if any(not math.isfinite(value) or not 0.0 <= value <= 0.25 for value in result):
        raise ValueError("computed pixel density proxy is outside [0, 0.25]")
    return result[0] if pixels.ndim == 3 else result


def route_preencoder_content_budget(
    *,
    query_simple_proxy: float | None,
    pixel_density_score: float | None,
    config: PreencoderBudgetConfig = PreencoderBudgetConfig(),
) -> dict[str, Any]:
    """Choose a request-level candidate ratio before placeholder expansion.

    Missing inputs fail safe to keep-all. The returned candidate can only
    affect serving when the current-backbone calibration flag is explicit.
    """
    for name, value in (
        ("query_simple_proxy", query_simple_proxy),
        ("pixel_density_score", pixel_density_score),
    ):
        if value is not None and not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite or None")
    if query_simple_proxy is not None and not 0.0 <= float(query_simple_proxy) <= 1.0:
        raise ValueError("query_simple_proxy must be in [0, 1] or None")
    if pixel_density_score is not None and not 0.0 <= float(pixel_density_score) <= 0.25:
        raise ValueError("pixel_density_score must be in [0, 0.25] or None")

    if query_simple_proxy is None or pixel_density_score is None:
        route = "insufficient_signals"
        candidate_keep_ratio = 1.0
        reason = "missing pre-E signal; preserve all visual tokens"
    elif float(pixel_density_score) >= config.pixel_dense_threshold:
        route = "dense_pixel_bypass"
        candidate_keep_ratio = 1.0
        reason = "pixel-density proxy requests a no-pruning bypass"
    elif float(query_simple_proxy) >= config.query_simple_threshold:
        route = "aggressive_proxy"
        candidate_keep_ratio = config.aggressive_keep_ratio
        reason = "low pixel-density proxy and lexically simple query"
    else:
        route = "conservative_proxy"
        candidate_keep_ratio = config.conservative_keep_ratio
        reason = "query is not lexically simple enough for aggressive candidate budget"

    active = bool(config.calibrated_for_current_backbone)
    return {
        "route": route,
        "candidate_keep_ratio": candidate_keep_ratio,
        "effective_keep_ratio": candidate_keep_ratio if active else 1.0,
        "active": active,
        "calibration_id": config.calibration_id,
        "query_simple_proxy": query_simple_proxy,
        "pixel_density_score": pixel_density_score,
        "query_simple_threshold": config.query_simple_threshold,
        "pixel_dense_threshold": config.pixel_dense_threshold,
        "reason": reason if active else reason + "; uncalibrated, serving remains keep-all",
        "signal_scope": "lexical query proxy + raw-pixel spatial variance; not ModernBERT QCR or embedding DIG",
    }


def choose_joint_preencoder_keep_count(
    *,
    full_token_count: int,
    safe_floor_count: int,
    candidate_keep_counts: Sequence[int],
    content_keep_ratio_cap: float,
    projected_text_impact_ms: Mapping[int, float],
    projected_image_ttft_ms: Mapping[int, float],
    text_slack_ms: float,
    image_slack_ms: float,
) -> dict[str, Any]:
    """Apply content cap and live D/I slack without crossing the quality floor."""
    if full_token_count < 1 or not 1 <= safe_floor_count <= full_token_count:
        raise ValueError("safe_floor_count must be within [1, full_token_count]")
    ratio = float(content_keep_ratio_cap)
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError("content_keep_ratio_cap must be finite and in (0, 1]")
    cap = min(full_token_count, max(1, math.ceil(full_token_count * ratio)))
    bounded = [int(k) for k in candidate_keep_counts if safe_floor_count <= int(k) <= cap]
    if safe_floor_count not in bounded:
        bounded.append(safe_floor_count)
    result = choose_deadline_constrained_budget(
        safe_floor_count=safe_floor_count,
        candidate_keep_counts=bounded,
        projected_text_impact_ms=projected_text_impact_ms,
        projected_image_ttft_ms=projected_image_ttft_ms,
        text_slack_ms=text_slack_ms,
        image_slack_ms=image_slack_ms,
    )
    result.update(
        {
            "full_token_count": full_token_count,
            "content_keep_ratio_cap": ratio,
            "content_budget_cap_count": cap,
            "content_cap_below_quality_floor": cap < safe_floor_count,
            "candidate_counts_within_content_cap": sorted(set(bounded)),
        }
    )
    return result


def request_ratio_for_exact_keep_count(full_token_count: int, keep_count: int) -> float:
    """Encode exact integer K through the current ratio-only API.

    The processor uses ``ceil(N * ratio)``. ``K/N`` can round upward by one
    ULP (for example N=25, K=7), so use the midpoint of the interval that
    maps to K. Return 1.0 for keep-all to preserve the established path.
    """
    if full_token_count < 1 or not 1 <= keep_count <= full_token_count:
        raise ValueError("keep_count must be within [1, full_token_count]")
    if keep_count == full_token_count:
        return 1.0
    ratio = (float(keep_count) - 0.5) / float(full_token_count)
    resolved = math.ceil(full_token_count * ratio)
    if resolved != keep_count:
        raise ArithmeticError(
            f"ratio {ratio!r} maps N={full_token_count} to K={resolved}, expected {keep_count}"
        )
    return ratio


def single_image_mm_processor_kwargs(full_token_count: int, keep_count: int) -> dict[str, Any]:
    """Build the request kwargs consumed before placeholder expansion in E."""
    ratio = request_ratio_for_exact_keep_count(full_token_count, keep_count)
    return {"images_kwargs": {"e_keep_ratio": ratio}}


def route_content_budget(
    *,
    qcr_simple_probability: float | None,
    dig_score: float | None,
    config: ContentPolicyConfig = ContentPolicyConfig(),
) -> dict[str, Any]:
    """Apply a dual-guard route while keeping uncalibrated operation fail-safe.

    The input is an actual QCR probability and a DIG-like embedding score.
    Heuristic query/image proxies must not be passed as calibrated probabilities.
    """
    if qcr_simple_probability is not None and (
        not math.isfinite(float(qcr_simple_probability)) or not 0.0 <= float(qcr_simple_probability) <= 1.0
    ):
        raise ValueError("qcr_simple_probability must be in [0, 1] or None")
    if dig_score is not None and not math.isfinite(float(dig_score)):
        raise ValueError("dig_score must be finite or None")

    if dig_score is None or qcr_simple_probability is None:
        route = "insufficient_signals"
        candidate_keep_ratio = 1.0
        reason = "missing guard signal; preserve all visual tokens"
    elif float(dig_score) < config.dig_dense_threshold:
        route = "dense_resolution_bypass"
        candidate_keep_ratio = 1.0
        reason = "DIG veto; bypass token pruning (resolution cap is a separate actuator)"
    elif float(qcr_simple_probability) >= config.qcr_simple_threshold:
        route = "aggressive"
        candidate_keep_ratio = 1.0 - config.aggressive_drop_ratio
        reason = "sparse image and high-confidence simple query"
    else:
        route = "conservative"
        candidate_keep_ratio = 1.0 - config.conservative_drop_ratio
        reason = "query is complex or below the conservative confidence threshold"

    active = bool(config.calibrated_for_current_backbone)
    return {
        "route": route,
        "candidate_keep_ratio": candidate_keep_ratio,
        "effective_keep_ratio": candidate_keep_ratio if active else 1.0,
        "active": active,
        "calibration_id": config.calibration_id,
        "qcr_simple_probability": qcr_simple_probability,
        "dig_score": dig_score,
        "qcr_threshold": config.qcr_simple_threshold,
        "dig_dense_threshold": config.dig_dense_threshold,
        "reason": reason if active else reason + "; current-backbone calibration is absent, so the serving actuator remains at keep-all",
    }


def dig_channel_variance_score(features: Any) -> float:
    """Compute the paper-shaped mean channel variance from an [N, D] tensor.

    The 0.55 paper threshold is not applied here. The returned scale must be
    checked on this model revision and calibration split before routing.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError("PyTorch is required to score vision embeddings") from exc
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError("features must be a torch.Tensor with shape [num_patches, hidden_size]")
    if features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError("features need at least two patches and one channel")
    score = features.detach().float().var(dim=0, unbiased=False).mean()
    result = float(score.item())
    if not math.isfinite(result):
        raise ValueError("computed DIG score is not finite")
    return result


_WORD_RE = re.compile(r"[\w]+", flags=re.UNICODE)
_COMPLEXITY_TERMS = {
    "compare", "versus", "difference", "count", "how", "many", "which", "locate",
    "identify", "read", "transcribe", "ocr", "text", "table", "chart", "graph",
    "document", "receipt", "invoice", "calculate", "trend", "before", "after",
    "relationship", "explain", "why", "evidence", "detail", "specific",
}


def query_complexity_proxy(query: str) -> dict[str, Any]:
    """Return transparent lexical features; this is not a QCR probability."""
    words = _WORD_RE.findall(query.casefold())
    unique_terms = sorted(set(words) & _COMPLEXITY_TERMS)
    digit_count = sum(ch.isdigit() for ch in query)
    clause_markers = sum(query.count(mark) for mark in (",", ";", "?", " and ", " or "))
    length_signal = min(len(words) / 40.0, 1.0)
    term_signal = min(len(unique_terms) / 5.0, 1.0)
    numeric_signal = min(digit_count / 8.0, 1.0)
    clause_signal = min(clause_markers / 4.0, 1.0)
    score = min(1.0, 0.30 * length_signal + 0.35 * term_signal + 0.20 * numeric_signal + 0.15 * clause_signal)
    return {
        "query_complexity_proxy": score,
        "query_simple_proxy": 1.0 - score,
        "word_count": len(words),
        "matched_terms": unique_terms,
        "digit_count": digit_count,
        "clause_markers": clause_markers,
        "calibrated_probability": False,
        "limitation": "lexical engineering proxy only; not the ModernBERT QCR and not quality-validated",
    }


def choose_deadline_constrained_budget(
    *,
    safe_floor_count: int,
    candidate_keep_counts: Sequence[int],
    projected_text_impact_ms: Mapping[int, float],
    projected_image_ttft_ms: Mapping[int, float],
    text_slack_ms: float,
    image_slack_ms: float,
) -> dict[str, Any]:
    """Choose the largest quality-safe candidate predicted to meet both slacks.

    The caller must provide calibrated cost curves. The floor is never lowered
    when deadlines are infeasible; the result reports the predicted miss.
    """
    if safe_floor_count < 1:
        raise ValueError("safe_floor_count must be >= 1")
    if text_slack_ms < 0 or image_slack_ms < 0:
        raise ValueError("deadline slacks must be non-negative")
    candidates = sorted({int(k) for k in candidate_keep_counts if int(k) >= safe_floor_count})
    if safe_floor_count not in candidates:
        candidates.insert(0, safe_floor_count)
    if not candidates:
        candidates = [safe_floor_count]
    missing = [k for k in candidates if k not in projected_text_impact_ms or k not in projected_image_ttft_ms]
    if missing:
        raise ValueError(f"cost curves are missing candidate keep counts: {missing}")
    for k in candidates:
        for metric, value in (("text", projected_text_impact_ms[k]), ("image", projected_image_ttft_ms[k])):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{metric} projected cost for K={k} must be finite and non-negative")

    feasible = [
        k for k in candidates
        if float(projected_text_impact_ms[k]) <= text_slack_ms
        and float(projected_image_ttft_ms[k]) <= image_slack_ms
    ]
    chosen = max(feasible) if feasible else safe_floor_count
    text_overrun = max(0.0, float(projected_text_impact_ms[chosen]) - text_slack_ms)
    image_overrun = max(0.0, float(projected_image_ttft_ms[chosen]) - image_slack_ms)
    return {
        "chosen_keep_count": chosen,
        "safe_floor_count": safe_floor_count,
        "deadline_feasible": bool(feasible),
        "feasible_keep_counts": feasible,
        "text_slack_ms": text_slack_ms,
        "image_slack_ms": image_slack_ms,
        "projected_text_impact_ms": float(projected_text_impact_ms[chosen]),
        "projected_image_ttft_ms": float(projected_image_ttft_ms[chosen]),
        "text_overrun_ms": text_overrun,
        "image_overrun_ms": image_overrun,
        "quality_floor_preserved": chosen >= safe_floor_count,
        "limitation": "candidate floor and cost curves require held-out quality and runtime calibration",
    }


def self_check() -> dict[str, Any]:
    uncalibrated = ContentPolicyConfig()
    assert route_content_budget(qcr_simple_probability=0.99, dig_score=0.9, config=uncalibrated)["effective_keep_ratio"] == 1.0
    calibrated = ContentPolicyConfig(calibrated_for_current_backbone=True, calibration_id="synthetic-self-check")
    aggressive = route_content_budget(qcr_simple_probability=0.99, dig_score=0.9, config=calibrated)
    conservative = route_content_budget(qcr_simple_probability=0.4, dig_score=0.9, config=calibrated)
    dense = route_content_budget(qcr_simple_probability=0.99, dig_score=0.2, config=calibrated)
    missing = route_content_budget(qcr_simple_probability=None, dig_score=0.9, config=calibrated)
    assert aggressive["route"] == "aggressive" and aggressive["effective_keep_ratio"] < conservative["effective_keep_ratio"]
    assert conservative["route"] == "conservative" and conservative["effective_keep_ratio"] == 1.0 - 0.667
    assert dense["route"] == "dense_resolution_bypass" and dense["effective_keep_ratio"] == 1.0
    assert missing["route"] == "insufficient_signals" and missing["effective_keep_ratio"] == 1.0

    selected = choose_deadline_constrained_budget(
        safe_floor_count=288,
        candidate_keep_counts=[144, 288, 432, 576],
        projected_text_impact_ms={144: 0.4, 288: 0.8, 432: 1.3, 576: 2.5},
        projected_image_ttft_ms={144: 0.7, 288: 1.1, 432: 1.7, 576: 2.3},
        text_slack_ms=1.5,
        image_slack_ms=2.0,
    )
    assert selected["chosen_keep_count"] == 432 and selected["deadline_feasible"]
    infeasible = choose_deadline_constrained_budget(
        safe_floor_count=288,
        candidate_keep_counts=[144, 288, 432],
        projected_text_impact_ms={144: 0.4, 288: 0.8, 432: 1.3},
        projected_image_ttft_ms={144: 0.7, 288: 1.1, 432: 1.7},
        text_slack_ms=0.6,
        image_slack_ms=0.5,
    )
    assert infeasible["chosen_keep_count"] == 288 and not infeasible["deadline_feasible"]
    assert infeasible["quality_floor_preserved"] and infeasible["image_overrun_ms"] > 0

    simple_query = query_complexity_proxy("What animal is in this photo?")
    dense_query = query_complexity_proxy("Read the chart, compare values 2020 and 2024, and explain the trend.")
    assert dense_query["query_complexity_proxy"] > simple_query["query_complexity_proxy"]

    proxy_cfg = PreencoderBudgetConfig()
    uncalibrated_proxy = route_preencoder_content_budget(
        query_simple_proxy=0.95, pixel_density_score=0.01, config=proxy_cfg
    )
    proxy_cfg_enabled = PreencoderBudgetConfig(
        calibrated_for_current_backbone=True,
        calibration_id="synthetic-branch-check-only",
    )
    aggressive_proxy = route_preencoder_content_budget(
        query_simple_proxy=0.95, pixel_density_score=0.01, config=proxy_cfg_enabled
    )
    dense_proxy = route_preencoder_content_budget(
        query_simple_proxy=0.95, pixel_density_score=0.10, config=proxy_cfg_enabled
    )
    missing_proxy = route_preencoder_content_budget(
        query_simple_proxy=0.95, pixel_density_score=None, config=proxy_cfg_enabled
    )
    assert uncalibrated_proxy["effective_keep_ratio"] == 1.0 and not uncalibrated_proxy["active"]
    assert aggressive_proxy["route"] == "aggressive_proxy" and aggressive_proxy["effective_keep_ratio"] == 0.222
    assert dense_proxy["route"] == "dense_pixel_bypass" and dense_proxy["effective_keep_ratio"] == 1.0
    assert missing_proxy["route"] == "insufficient_signals" and missing_proxy["effective_keep_ratio"] == 1.0

    joint = choose_joint_preencoder_keep_count(
        full_token_count=576,
        safe_floor_count=192,
        candidate_keep_counts=[144, 192, 256, 320, 384, 512, 576],
        content_keep_ratio_cap=0.6,
        projected_text_impact_ms={192: 0.5, 256: 0.7, 320: 0.9, 384: 1.1, 512: 1.5, 576: 2.0},
        projected_image_ttft_ms={192: 0.6, 256: 0.8, 320: 1.0, 384: 1.2, 512: 1.6, 576: 2.2},
        text_slack_ms=1.0,
        image_slack_ms=1.1,
    )
    assert joint["chosen_keep_count"] == 320
    assert joint["chosen_keep_count"] <= joint["content_budget_cap_count"]
    below_floor = choose_joint_preencoder_keep_count(
        full_token_count=256,
        safe_floor_count=192,
        candidate_keep_counts=[64, 128],
        content_keep_ratio_cap=0.25,
        projected_text_impact_ms={192: 0.9},
        projected_image_ttft_ms={192: 1.1},
        text_slack_ms=0.1,
        image_slack_ms=0.1,
    )
    assert below_floor["chosen_keep_count"] == 192
    assert below_floor["content_cap_below_quality_floor"] and not below_floor["deadline_feasible"]
    ratio_cases = [(25, 7), (29, 15), (256, 128), (576, 383), (576, 576)]
    for full_count, keep_count in ratio_cases:
        ratio = request_ratio_for_exact_keep_count(full_count, keep_count)
        assert math.ceil(full_count * ratio) == keep_count
        kwargs = single_image_mm_processor_kwargs(full_count, keep_count)
        assert math.ceil(full_count * kwargs["images_kwargs"]["e_keep_ratio"]) == keep_count
    for full_count in range(1, 2049):
        for keep_count in range(1, full_count + 1):
            ratio = request_ratio_for_exact_keep_count(full_count, keep_count)
            assert math.ceil(full_count * ratio) == keep_count

    dig_check: dict[str, Any] = {"torch_available": False}
    try:
        import torch

        low = dig_channel_variance_score(torch.zeros((8, 4)))
        high_features = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        high = dig_channel_variance_score(high_features)
        assert low == 0.0 and high > low
        flat_pixels = torch.full((3, 16, 16), 0.5)
        textured_pixels = torch.rand((3, 16, 16), generator=torch.Generator().manual_seed(7))
        flat_pixel_score = raw_pixel_density_proxy(flat_pixels)
        textured_pixel_score = raw_pixel_density_proxy(textured_pixels)
        assert flat_pixel_score == 0.0 and textured_pixel_score > flat_pixel_score
        dig_check = {
            "torch_available": True,
            "low_variance_score": low,
            "high_variance_score": high,
            "flat_pixel_score": flat_pixel_score,
            "textured_pixel_score": textured_pixel_score,
        }
    except ImportError:
        pass

    return {
        "valid": True,
        "checks": {
            "uncalibrated_policy_keeps_all": True,
            "dual_guard_aggressive_conservative_dense_routes": True,
            "missing_signal_fails_safe": True,
            "deadline_selector_respects_quality_floor": True,
            "query_proxy_ordering": True,
            "dig_variance_score": dig_check,
            "preencoder_proxy_dual_guard_and_fail_safe": True,
            "joint_budget_obeys_content_cap_deadlines_and_quality_floor": True,
            "integer_keep_count_ratio_mapping_exhaustive_1_to_2048": True,
        },
        "paper_reference": PAPER_REFERENCE,
        "limitations": [
            "No current-backbone QCR training or calibration is included.",
            "SieveServe paper thresholds are reference values only and are not enabled by default.",
            "The module is not integrated with vLLM's post-E/pre-P request compaction path.",
            "The raw-pixel variance proxy is not SieveServe DIG and has not been calibrated against Qwen3-VL-2B quality.",
            "Synthetic self-check is not a quality, latency, or goodput result.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if not args.self_check:
        parser.error("select --self-check; this exploratory module does not route serving traffic")
    result = self_check()
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded, encoding="utf-8")
    print(json.dumps({"valid": result["valid"], "checks": result["checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
