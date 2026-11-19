"""Experimental Sage1 kernel with 64-row compute tiles and Q128 scales.

This module is intentionally kept outside the SGLang runtime path. Quantization,
the K traversal, and the arithmetic in SageAttention's inner loop stay pinned to
the installed SageAttention 1.0.6 implementation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sageattention.attn_qk_int8_per_block import _attn_fwd_inner


@triton.jit
def _attn_fwd_qtile64(
    Q,
    K,
    V,
    Q_scale,
    K_scale,
    Out,
    stride_qz,
    stride_qh,
    stride_qn,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_on,
    qo_len,
    kv_len,
    H: tl.constexpr,
    num_kv_groups: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_QUANT_BLOCK: tl.constexpr,
    STAGE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    # Scale storage remains grouped by 128 Q rows even though each CTA handles
    # 64 rows. Two adjacent CTAs therefore reuse one Q scale block.
    q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, Q_QUANT_BLOCK)
    q_quant_block_index = (start_m * BLOCK_M) // Q_QUANT_BLOCK
    k_scale_offset = (
        (off_z * (H // num_kv_groups) + off_h // num_kv_groups)
        * tl.cdiv(kv_len, BLOCK_N)
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = (
        Q
        + (off_z * stride_qz + off_h * stride_qh)
        + offs_m[:, None] * stride_qn
        + offs_k[None, :]
    )
    Q_scale_ptr = Q_scale + q_scale_offset + q_quant_block_index
    K_ptrs = (
        K
        + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh)
        + offs_n[None, :] * stride_kn
        + offs_k[:, None]
    )
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = (
        V
        + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh)
        + offs_n[:, None] * stride_vn
        + offs_k[None, :]
    )
    O_block_ptr = (
        Out
        + (off_z * stride_oz + off_h * stride_oh)
        + offs_m[:, None] * stride_on
        + offs_k[None, :]
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale_ptr)
    acc, l_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        q_scale,
        kv_len,
        K_ptrs,
        K_scale_ptr,
        V_ptrs,
        stride_kn,
        stride_vn,
        start_m,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,
        4 - STAGE,
        offs_m,
        offs_n,
    )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=offs_m[:, None] < qo_len)


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    tensor_layout: str = "HND",
    output_dtype: torch.dtype = torch.float16,
    *,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Run the isolated Q-tile-64 candidate for HND or NHD tensors."""
    if num_warps not in (4, 8):
        raise ValueError(f"unsupported num_warps={num_warps}")
    if num_stages not in (3, 4):
        raise ValueError(f"unsupported num_stages={num_stages}")

    block_m = 64
    block_n = 64
    q_quant_block = 128
    stage = 1  # preserve Sage1's non-causal inner-loop path
    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        strides_q = (q.stride(0), q.stride(1), q.stride(2))
        strides_k = (k.stride(0), k.stride(1), k.stride(2))
        strides_v = (v.stride(0), v.stride(1), v.stride(2))
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        strides_q = (q.stride(0), q.stride(2), q.stride(1))
        strides_k = (k.stride(0), k.stride(2), k.stride(1))
        strides_v = (v.stride(0), v.stride(2), v.stride(1))
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")

    out = torch.empty(q.shape, dtype=output_dtype, device=q.device)
    strides_out = (
        (out.stride(0), out.stride(2), out.stride(1))
        if tensor_layout == "NHD"
        else (out.stride(0), out.stride(1), out.stride(2))
    )
    grid = (triton.cdiv(qo_len, block_m), h_qo, b)
    _attn_fwd_qtile64[grid](
        q,
        k,
        v,
        q_scale,
        k_scale,
        out,
        *strides_q,
        *strides_k,
        *strides_v,
        *strides_out,
        qo_len,
        kv_len,
        h_qo,
        h_qo // h_kv,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        Q_QUANT_BLOCK=q_quant_block,
        STAGE=stage,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
