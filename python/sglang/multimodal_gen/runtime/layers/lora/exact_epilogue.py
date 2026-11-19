# SPDX-License-Identifier: Apache-2.0
"""Opt-in CUDA BF16 LoRA epilogue preserving each eager rounding boundary."""

import torch
import triton
import triton.language as tl

from sglang.kernels.numerics import mul_rn_f32, round_bf16_to_fp32


@triton.jit
def _exact_lora_add_kernel(
    base,
    delta,
    output,
    size: tl.constexpr,
    scales: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    valid = offsets < size
    value = tl.load(delta + offsets, valid, other=0).to(tl.float32)
    for index in tl.static_range(len(scales)):
        # Do not combine scales: eager rounds after every multiplication.
        value = (value * scales[index]).to(tl.bfloat16).to(tl.float32)
    original = tl.load(base + offsets, valid, other=0).to(tl.float32)
    tl.store(output + offsets, original + value, valid)


def exact_lora_add(
    base: torch.Tensor, delta: torch.Tensor, scales: tuple[float, ...]
) -> torch.Tensor:
    """Return a fresh contiguous result; caller checks CUDA/BF16/layout/grad."""
    output = torch.empty_like(base)
    if base.numel():
        _exact_lora_add_kernel[(triton.cdiv(base.numel(), 1024),)](
            base,
            delta,
            output,
            base.numel(),
            scales,
            1024,
            enable_fp_fusion=False,
        )
    return output


@triton.jit
def _exact_lora_silu_mul_kernel(
    base,
    delta,
    output,
    rows,
    width: tl.constexpr,
    scales: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * block + tl.arange(0, block)
    valid = (row < rows) & (cols < width)
    gate_offsets = row * (2 * width) + cols
    up_offsets = gate_offsets + width

    gate_delta = tl.load(delta + gate_offsets, valid, other=0).to(tl.float32)
    up_delta = tl.load(delta + up_offsets, valid, other=0).to(tl.float32)
    for index in tl.static_range(len(scales)):
        gate_delta = (gate_delta * scales[index]).to(tl.bfloat16).to(tl.float32)
        up_delta = (up_delta * scales[index]).to(tl.bfloat16).to(tl.float32)

    gate_base = tl.load(base + gate_offsets, valid, other=0).to(tl.float32)
    up_base = tl.load(base + up_offsets, valid, other=0).to(tl.float32)
    gate = (gate_base + gate_delta).to(tl.bfloat16).to(tl.float32)
    up = (up_base + up_delta).to(tl.bfloat16).to(tl.float32)
    activated = round_bf16_to_fp32(gate * tl.sigmoid(gate))
    tl.store(output + row * width + cols, activated * up, valid)


def exact_lora_silu_mul(
    base: torch.Tensor, delta: torch.Tensor, scales: tuple[float, ...]
) -> torch.Tensor:
    """Apply exact BF16 LoRA addition and rounded SwiGLU in one kernel."""
    if not (
        base.is_cuda
        and delta.is_cuda
        and base.device == delta.device
        and base.dtype == delta.dtype == torch.bfloat16
        and base.shape == delta.shape
        and base.is_contiguous()
        and delta.is_contiguous()
        and base.ndim >= 2
        and base.shape[-1] > 0
        and base.shape[-1] % 2 == 0
        and base.numel() > 0
    ):
        raise RuntimeError("invalid input for exact_lora_silu_mul")
    width = base.shape[-1] // 2
    rows = base.numel() // base.shape[-1]
    output_shape = (*base.shape[:-1], width)
    output = torch.empty(output_shape, dtype=base.dtype, device=base.device)
    _exact_lora_silu_mul_kernel[(rows, triton.cdiv(width, 1024))](
        base,
        delta,
        output,
        rows,
        width,
        scales,
        1024,
        enable_fp_fusion=False,
    )
    return output

@triton.jit
def _exact_lora_gate_add_kernel(
    base,
    delta,
    residual,
    gate,
    indices,
    output,
    hidden_size,
    stride_base_row,
    stride_delta_row,
    stride_residual_row,
    stride_gate_row,
    stride_indices,
    stride_output_row,
    scales: tl.constexpr,
    GATE_IS_BF16: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = columns < hidden_size
    index = tl.load(indices + row * stride_indices)

    delta_value = tl.load(
        delta + row * stride_delta_row + columns, valid, other=0
    ).to(tl.float32)
    for scale_index in tl.static_range(len(scales)):
        delta_value = (
            delta_value * scales[scale_index]
        ).to(tl.bfloat16).to(tl.float32)

    base_value = tl.load(
        base + row * stride_base_row + columns, valid, other=0
    ).to(tl.float32)
    projected = round_bf16_to_fp32(base_value + delta_value)

    gate_value = tl.load(
        gate + index * stride_gate_row + columns, valid, other=0
    ).to(tl.float32)
    if GATE_IS_BF16:
        gated = round_bf16_to_fp32(gate_value * projected)
    else:
        gated = mul_rn_f32(gate_value, projected)

    residual_value = tl.load(
        residual + row * stride_residual_row + columns, valid, other=0
    ).to(tl.float32)
    tl.store(
        output + row * stride_output_row + columns,
        residual_value + gated,
        valid,
    )



_EXACT_LORA_GATE_BENCHMARK_DONE = False


def _maybe_run_exact_lora_gate_microbenchmark(
    base: torch.Tensor,
    delta: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    indices: torch.Tensor,
    scales: tuple[float, ...],
) -> None:
    """One-shot CUDA Event attribution on real model activations, opt-in by env."""
    global _EXACT_LORA_GATE_BENCHMARK_DONE
    if _EXACT_LORA_GATE_BENCHMARK_DONE:
        return

    import json
    import os
    import pathlib
    import statistics

    output_path = os.environ.get("SGLANG_EXACT_LORA_GATE_MICROBENCH_PATH")
    if os.environ.get("SGLANG_EXACT_LORA_GATE_MICROBENCH") != "1" or not output_path:
        return
    _EXACT_LORA_GATE_BENCHMARK_DONE = True

    from sglang.kernels.ops.diffusion.modulate.indexed_modulation_triton import (
        indexed_gate_bf16_,
        indexed_gate_fp32_,
    )

    rows = min(4096, base.shape[0])
    sample_base = base[:rows]
    sample_delta = delta[:rows]
    sample_residual = residual[:rows]
    sample_indices = indices[:rows]
    reference_residual = sample_residual.clone()
    candidate_residual = sample_residual.clone()
    original_residual = sample_residual.clone()
    gate_fn = indexed_gate_bf16_ if gate.dtype == torch.bfloat16 else indexed_gate_fp32_

    projected = exact_lora_add(sample_base, sample_delta, scales)
    if gate.dtype == torch.bfloat16:
        expected = indexed_gate_bf16_(
            sample_residual.clone(), gate, projected, sample_indices
        )
    else:
        expected = indexed_gate_fp32_(
            sample_residual.clone(), gate, projected, sample_indices
        )
    actual = exact_lora_gate_add(
        sample_base,
        sample_delta,
        sample_residual,
        gate,
        sample_indices,
        scales,
        inplace=False,
    )
    torch.cuda.synchronize(base.device)
    if not torch.equal(expected, actual):
        raise RuntimeError("real activation exact LoRA gate check failed")

    def reference_inplace():
        projected_value = exact_lora_add(sample_base, sample_delta, scales)
        return gate_fn(
            reference_residual, gate, projected_value, sample_indices
        )

    def candidate_inplace():
        return exact_lora_gate_add(
            sample_base,
            sample_delta,
            candidate_residual,
            gate,
            sample_indices,
            scales,
            inplace=True,
        )

    def timed(fn, work):
        work.copy_(original_residual)
        torch.cuda.synchronize(base.device)
        for _ in range(2):
            fn()
        torch.cuda.synchronize(base.device)
        work.copy_(original_residual)
        torch.cuda.synchronize(base.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(8):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / 8.0

    measurements = {"reference_two_kernels_ms": [], "fused_one_kernel_ms": []}
    funcs = {
        "reference_two_kernels_ms": (reference_inplace, reference_residual),
        "fused_one_kernel_ms": (candidate_inplace, candidate_residual),
    }
    for round_index in range(3):
        names = list(funcs)
        if round_index % 2:
            names.reverse()
        for name in names:
            fn, work = funcs[name]
            measurements[name].append(timed(fn, work))

    result = {
        "source": "4096-row slice of real FC2 activations from the first opt-in H3 request",
        "full_shape": list(base.shape),
        "measured_shape": [rows, base.shape[1]],
        "gate_dtype": str(gate.dtype),
        "scales": list(scales),
        "exact_output_match": True,
        "rounds": 3,
        "repeats_per_round": 8,
        "per_round_ms_per_call": measurements,
        "median_ms_per_call": {
            name: statistics.median(values) for name, values in measurements.items()
        },
    }
    result["median_reduction_pct"] = 100.0 * (
        1.0
        - result["median_ms_per_call"]["fused_one_kernel_ms"]
        / result["median_ms_per_call"]["reference_two_kernels_ms"]
    )
    destination = pathlib.Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.write_text(json.dumps(result, indent=2) + "\n")


def exact_lora_gate_add(
    base: torch.Tensor,
    delta: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    indices: torch.Tensor,
    scales: tuple[float, ...],
    *,
    inplace: bool = False,
) -> torch.Tensor:
    """Fuse exact BF16 LoRA addition with H3's indexed gated residual."""
    if not (
        base.is_cuda
        and delta.is_cuda
        and residual.is_cuda
        and gate.is_cuda
        and indices.is_cuda
        and base.device == delta.device == residual.device == gate.device == indices.device
        and base.dtype == delta.dtype == residual.dtype == torch.bfloat16
        and gate.dtype in (torch.bfloat16, torch.float32)
        and base.shape == delta.shape == residual.shape
        and base.ndim == 2
        and base.shape[0] == indices.numel()
        and gate.ndim == 2
        and gate.shape[1] == base.shape[1]
        and indices.ndim == 1
        and indices.dtype in (torch.int32, torch.int64)
        and base.is_contiguous()
        and delta.is_contiguous()
        and residual.is_contiguous()
        and gate.stride(-1) == 1
        and base.shape[1] > 0
    ):
        raise RuntimeError("invalid input for exact_lora_gate_add")

    _maybe_run_exact_lora_gate_microbenchmark(
        base, delta, residual, gate, indices, scales
    )
    output = residual if inplace else torch.empty_like(residual)
    rows, hidden_size = base.shape
    if rows == 0:
        return output

    block_n = min(2048, triton.next_power_of_2(hidden_size))
    _exact_lora_gate_add_kernel[(rows, triton.cdiv(hidden_size, block_n))](
        base,
        delta,
        residual,
        gate,
        indices,
        output,
        hidden_size,
        base.stride(0),
        delta.stride(0),
        residual.stride(0),
        gate.stride(0),
        indices.stride(0),
        output.stride(0),
        scales,
        gate.dtype == torch.bfloat16,
        block_n,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
