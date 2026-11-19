import json
import os
import statistics
from pathlib import Path

import torch
import triton

from sglang.kernels.ops.activation.activation import (
    silu_and_mul_with_activation_rounding,
)
from sglang.multimodal_gen.runtime.layers.lora.exact_epilogue import (
    exact_lora_add,
    exact_lora_silu_mul,
)


ROWS = 37760
PACKED_WIDTH = 28672
SCALES = (0.7, 0.6)
WARMUP = 5
SAMPLES = 30
ROUNDS = 3
OUTPUT = Path(
    "/home/chx/projects/sglang-h3-exact-lora-20261029/experiments/"
    "minimax-h3-sm89-kernel-20261030/results/"
    "fc1-lora-silu-fusion-gpu1.json"
)


def baseline(base: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    merged = exact_lora_add(base, delta, SCALES)
    return silu_and_mul_with_activation_rounding(merged)


def candidate(base: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return exact_lora_silu_mul(base, delta, SCALES)


def sample(fn, base, delta):
    for _ in range(WARMUP):
        output = fn(base, delta)
        del output
    torch.cuda.synchronize()

    samples_ms = []
    for _ in range(SAMPLES):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = fn(base, delta)
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end))
        del output
    return samples_ms


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    torch.manual_seed(5151)
    base = torch.randn(
        (ROWS, PACKED_WIDTH), device="cuda", dtype=torch.bfloat16
    )
    delta = torch.randn_like(base)

    expected = baseline(base, delta)
    actual = candidate(base, delta)
    torch.cuda.synchronize()
    equal = torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    mismatch_count = int((actual.view(torch.int16) != expected.view(torch.int16)).sum())
    max_abs = float((actual.float() - expected.float()).abs().max())
    result = {
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_home": os.environ.get("CUDA_HOME"),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "shape": [ROWS, PACKED_WIDTH],
        "dtype": "bfloat16",
        "scales": SCALES,
        "warmup": WARMUP,
        "samples_per_cell": SAMPLES,
        "abba_rounds": ROUNDS,
        "bitwise_equal": equal,
        "mismatch_count": mismatch_count,
        "max_abs": max_abs,
    }
    del expected, actual
    torch.cuda.synchronize()
    if not equal:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
        raise RuntimeError(f"candidate differs from two-pass reference: {result}")

    by_backend = {"baseline": [], "candidate": []}
    round_details = []
    for round_index in range(ROUNDS):
        order = (
            ("baseline", baseline),
            ("candidate", candidate),
            ("candidate", candidate),
            ("baseline", baseline),
        )
        cells = []
        for name, fn in order:
            samples_ms = sample(fn, base, delta)
            by_backend[name].extend(samples_ms)
            cells.append(
                {
                    "backend": name,
                    "median_ms": statistics.median(samples_ms),
                    "p90_ms": sorted(samples_ms)[int(0.9 * (len(samples_ms) - 1))],
                }
            )
        round_details.append({"round": round_index + 1, "abba_cells": cells})

    baseline_median = statistics.median(by_backend["baseline"])
    candidate_median = statistics.median(by_backend["candidate"])
    result.update(
        {
            "baseline_median_ms": baseline_median,
            "candidate_median_ms": candidate_median,
            "speedup": baseline_median / candidate_median,
            "latency_reduction_pct": (
                1.0 - candidate_median / baseline_median
            )
            * 100.0,
            "baseline_samples_ms": by_backend["baseline"],
            "candidate_samples_ms": by_backend["candidate"],
            "round_details": round_details,
        }
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
