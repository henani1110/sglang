# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.lora.linear import (
    LinearWithLoRA,
    _can_add_lora_delta_in_gemm,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context


@pytest.mark.parametrize("enabled", [False, True])
def test_experimental_lora_delta_in_gemm_is_request_scoped(enabled):
    out = torch.empty((8, 96), dtype=torch.bfloat16)
    lora_a = torch.empty((16, 192), dtype=torch.bfloat16)
    lora_b = torch.empty((96, 16), dtype=torch.bfloat16)
    batch = SimpleNamespace(
        sampling_params=SimpleNamespace(
            quality="exact", experimental_lora_delta_in_gemm=enabled
        )
    )
    with set_forward_context(0, None, batch):
        actual = _can_add_lora_delta_in_gemm(out, lora_a, lora_b)
    assert actual is enabled


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("stacked", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "alpha, strength, scale",
    [(16, 1.0, 1.0), (8, 0.7, 0.6), (32, -1.3, 0.25), (16, 0.0, 1.0)],
)
def test_integrated_exact_chain(monkeypatch, stacked, dtype, alpha, strength, scale):
    torch.manual_seed(1101)
    base = torch.nn.Linear(192, 96, dtype=torch.bfloat16, device="cuda")
    layer = LinearWithLoRA(base, lora_rank=16, lora_alpha=alpha)
    a_shape = (3, 16, 192) if stacked else (16, 192)
    b_shape = (3, 32, 16) if stacked else (96, 16)
    a = torch.randn(a_shape, device="cuda", dtype=dtype) * 0.05
    b = torch.randn(b_shape, device="cuda", dtype=dtype) * 0.05
    layer.set_lora_weights(a, b, strength=strength, merge_weights=False)
    x = torch.randn(257, 192, device="cuda", dtype=torch.bfloat16)
    batch = SimpleNamespace(
        sampling_params=SimpleNamespace(quality="exact"),
        runtime_lora_scale=scale,
    )
    with torch.inference_mode(), set_forward_context(0, None, batch):
        out = base(x)
        snapshot = out.clone()
        monkeypatch.setenv("SGLANG_DIFFUSION_EXACT_LORA_EPILOGUE", "0")
        expected = layer._add_lora_delta(out, x, scale)
        monkeypatch.setenv("SGLANG_DIFFUSION_EXACT_LORA_EPILOGUE", "1")
        actual = layer._add_lora_delta(out, x, scale)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(out, snapshot)  # Do not overwrite an aliased base result.
    assert actual.data_ptr() != out.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compiled_lora_forward_uses_opt_in_epilogue(monkeypatch):
    torch.manual_seed(1301)
    base = torch.nn.Linear(192, 96, dtype=torch.bfloat16, device="cuda")
    layer = LinearWithLoRA(base, lora_rank=16, lora_alpha=8)
    a = torch.randn(16, 192, device="cuda", dtype=torch.bfloat16) * 0.05
    b = torch.randn(96, 16, device="cuda", dtype=torch.bfloat16) * 0.05
    layer.set_lora_weights(a, b, strength=0.7, merge_weights=False)
    x = torch.randn(257, 192, device="cuda", dtype=torch.bfloat16)
    batch = SimpleNamespace(
        sampling_params=SimpleNamespace(quality="exact", exact_lora_epilogue=True),
        runtime_lora_scale=0.6,
    )
    with torch.inference_mode(), set_forward_context(0, None, batch):
        monkeypatch.setenv("SGLANG_DIFFUSION_EXACT_LORA_EPILOGUE", "0")
        expected = layer(x)
        monkeypatch.setenv("SGLANG_DIFFUSION_EXACT_LORA_EPILOGUE", "1")
        actual = layer(x)
    assert torch.equal(actual[0].view(torch.int16), expected[0].view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n", [0, 1, 1023, 1024, 1025, 65539])
@pytest.mark.parametrize("scales", [(), (0.7,), (0.5, 0.7, 0.6), (-1.3, 0.25)])
def test_rounding_and_mask(n, scales):
    from sglang.multimodal_gen.runtime.layers.lora.exact_epilogue import exact_lora_add

    torch.manual_seed(42)
    a = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    ref = b
    for scale in scales:
        ref = ref * scale
    ref = a + ref
    actual = exact_lora_add(a, b, scales)
    assert torch.equal(actual.view(torch.int16), ref.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_special_values():
    from sglang.multimodal_gen.runtime.layers.lora.exact_epilogue import exact_lora_add

    values = [0.0, -0.0, float("inf"), -float("inf"), 1e38, -1e38, 1e-38, -1e-38]
    a = torch.tensor(values, device="cuda", dtype=torch.bfloat16)
    b = a.flip(0)
    for scale in [0.7, 0.0, -1.3]:
        expected = a + b * scale
        actual = exact_lora_add(a, b, (scale,))
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("scales", [(), (0.7,), (0.5, 0.7, 0.6), (-1.3, 0.25)])
def test_exact_lora_silu_mul_matches_two_pass_chain(scales):
    from sglang.kernels.ops.activation.activation import (
        silu_and_mul_with_activation_rounding,
    )
    from sglang.multimodal_gen.runtime.layers.lora.exact_epilogue import (
        exact_lora_add,
        exact_lora_silu_mul,
    )

    torch.manual_seed(5151)
    base = torch.randn((19, 256), device="cuda", dtype=torch.bfloat16)
    delta = torch.randn_like(base)
    expected = silu_and_mul_with_activation_rounding(
        exact_lora_add(base, delta, scales)
    )
    actual = exact_lora_silu_mul(base, delta, scales)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
