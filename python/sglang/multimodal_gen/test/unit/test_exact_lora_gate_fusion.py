# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sglang.kernels.ops.diffusion.modulate.indexed_modulation_triton import (
    indexed_gate_bf16,
    indexed_gate_fp32,
)
from sglang.multimodal_gen.runtime.layers.lora.exact_epilogue import (
    exact_lora_add,
    exact_lora_gate_add,
)


@pytest.mark.parametrize("gate_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("inplace", [False, True])
def test_exact_lora_gate_add_matches_reference(gate_dtype, inplace):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(41)
    rows, hidden, gate_rows = 17, 257, 5
    base = torch.randn((rows, hidden), device="cuda", dtype=torch.bfloat16)
    delta = torch.randn_like(base)
    residual = torch.randn_like(base)
    gate = torch.randn((gate_rows, hidden), device="cuda", dtype=gate_dtype)
    indices = torch.randint(gate_rows, (rows,), device="cuda", dtype=torch.int64)
    scales = (0.7, 0.6)
    original_residual = residual.clone()

    projected = exact_lora_add(base, delta, scales)
    gate_fn = indexed_gate_bf16 if gate_dtype == torch.bfloat16 else indexed_gate_fp32
    expected = gate_fn(residual, gate, projected, indices)

    actual_residual = residual.clone() if inplace else residual
    actual = exact_lora_gate_add(
        base,
        delta,
        actual_residual,
        gate,
        indices,
        scales,
        inplace=inplace,
    )

    assert torch.equal(actual, expected)
    if inplace:
        assert actual.data_ptr() == actual_residual.data_ptr()
    else:
        assert torch.equal(residual, original_residual)
