#!/usr/bin/env python3
"""GeoPrune-style residual ranking primitives for Qwen3-VL E experiments.

This module implements the paper-described fixed-K deflation score only. The
paper's adaptive eigengap pseudocode is ambiguous, and mapping pre-merger ViT
patches to decoder-visible merge groups is not specified. Callers must freeze
those choices and must not label this module an exact paper reproduction.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import torch


@dataclass
class ResidualRanking:
    scores: torch.Tensor
    residual: torch.Tensor
    singular_values: list[float]


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    return vector / norm.clamp_min(torch.finfo(vector.dtype).eps)


@torch.no_grad()
def residual_l2_scores(
    features: torch.Tensor,
    *,
    deflation_k: int,
    power_iterations: int = 10,
    seed: int = 0,
) -> ResidualRanking:
    """Compute q_i after exactly `deflation_k` rank-one power-iteration steps.

    `features` must be ordered exactly as the last ViT block output entering
    the merger. Adaptive K is intentionally not inferred here. Work is done in
    float32 for numerical stability; the returned residual/scores stay on the
    input device.
    """
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("features must have non-empty [num_patches, hidden_size] shape")
    if deflation_k < 0 or deflation_k > 3:
        raise ValueError("deflation_k must be in [0, 3] for the paper's K_max")
    if power_iterations < 1:
        raise ValueError("power_iterations must be positive")
    if not torch.isfinite(features).all():
        raise ValueError("features contain NaN or Inf")

    work = features.to(dtype=torch.float32)
    residual = work.clone()
    generator = torch.Generator(device=work.device)
    generator.manual_seed(seed)
    singular_values: list[float] = []

    for _ in range(deflation_k):
        v = _normalize(torch.randn(
            (residual.shape[1],),
            dtype=residual.dtype,
            device=residual.device,
            generator=generator,
        ))
        for _ in range(power_iterations):
            u = _normalize(residual @ v)
            v = _normalize(residual.T @ u)
        u = _normalize(residual @ v)
        sigma = torch.dot(u, residual @ v).abs()
        sigma_value = float(sigma.item())
        if not math.isfinite(sigma_value) or sigma_value <= torch.finfo(residual.dtype).eps:
            break
        residual = residual - sigma * u[:, None] * v[None, :]
        singular_values.append(sigma_value)

    scores = torch.linalg.vector_norm(residual, ord=2, dim=1)
    return ResidualRanking(
        scores=scores,
        residual=residual,
        singular_values=singular_values,
    )


def merger_group_mean_scores(
    patch_scores: torch.Tensor,
    *,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Average patch scores in contiguous groups consumed by the Qwen merger.

    The Qwen3-VL merger concatenates `spatial_merge_size**2` consecutive
    pre-merger rows. This helper follows that merger-input order. It is an
    explicit engineering mapping, not specified by GeoPrune's paper.
    """
    if patch_scores.ndim != 1:
        raise ValueError("patch_scores must be a flat vector")
    if spatial_merge_size < 1:
        raise ValueError("spatial_merge_size must be positive")
    group_size = spatial_merge_size**2
    if patch_scores.numel() == 0 or patch_scores.numel() % group_size:
        raise ValueError("patch score count must be a positive multiple of merge group size")
    return patch_scores.reshape(-1, group_size).mean(dim=1)


def select_group_indices(group_scores: torch.Tensor, keep_count: int) -> torch.Tensor:
    """Select top scoring groups, then restore merger order for stable positions."""
    if group_scores.ndim != 1 or group_scores.numel() == 0:
        raise ValueError("group_scores must be a non-empty flat vector")
    if keep_count < 1 or keep_count > group_scores.numel():
        raise ValueError("keep_count must be between 1 and the number of merge groups")
    ranked = torch.argsort(group_scores, descending=True, stable=True)[:keep_count]
    return torch.sort(ranked).values


def _self_check() -> dict[str, object]:
    torch.manual_seed(1709)
    shared = torch.randn(1, 8)
    residual_component = torch.randn(16, 8) * 0.08
    features = torch.ones(16, 1) @ (shared * 10.0) + residual_component
    result = residual_l2_scores(features, deflation_k=1, power_iterations=10, seed=1710)
    u_ref, s_ref, vh_ref = torch.linalg.svd(features, full_matrices=False)
    exact_residual = features - s_ref[0] * u_ref[:, :1] * vh_ref[:1, :]
    exact_scores = torch.linalg.vector_norm(exact_residual, dim=1)
    max_reference_error = float((result.scores - exact_scores).abs().max().item())
    matches_svd_reference = torch.allclose(
        result.scores, exact_scores, atol=1e-4, rtol=1e-2
    )
    groups = merger_group_mean_scores(result.scores, spatial_merge_size=2)
    kept = select_group_indices(groups, keep_count=2)
    tie_probe = select_group_indices(
        torch.tensor([0.1, 0.8, 0.8, 0.2]), keep_count=2
    )
    valid = (
        len(result.singular_values) == 1
        and torch.isfinite(result.scores).all().item()
        and groups.numel() == 4
        and kept.numel() == 2
        and torch.all(kept[1:] > kept[:-1]).item()
        and torch.equal(tie_probe, torch.tensor([1, 2]))
        and matches_svd_reference
    )
    return {
        "schema_version": 1,
        "valid": bool(valid),
        "features_shape": list(features.shape),
        "deflation_k": 1,
        "power_iterations": 10,
        "estimated_singular_values": result.singular_values,
        "svd_reference_max_abs_score_error": max_reference_error,
        "matches_svd_reference": bool(matches_svd_reference),
        "merge_group_count": int(groups.numel()),
        "keep_count": int(kept.numel()),
        "keep_indices_in_merger_order": kept.tolist(),
        "stable_tie_probe_indices": tie_probe.tolist(),
        "limitations": [
            "CPU synthetic numerical smoke only; no Qwen model integration or answer-quality claim.",
            "Adaptive eigengap and pre-merger-to-decoder mapping remain explicit caller choices.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if not args.self_check:
        parser.error("only --self-check is exposed until the model handoff is integrated")
    result = _self_check()
    print(json.dumps(result, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
