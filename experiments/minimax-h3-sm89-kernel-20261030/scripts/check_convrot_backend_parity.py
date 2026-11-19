"""Compare the installed CUDA and Triton INT8 linear paths on one fixed input."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import torch

from comfy_kitchen.backends.triton.quantization import int8_linear as triton_int8_linear


def run(seed: int, m: int, n: int, k: int) -> dict:
    torch.manual_seed(seed)
    torch.cuda.set_device(0)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    weight_scale = torch.rand((n,), device="cuda", dtype=torch.float32)
    rows = []
    for convrot in (False, True):
        cuda_output = torch.ops.comfy_kitchen.int8_linear(
            x, weight, weight_scale, None, 2, convrot, 256
        )
        triton_output = triton_int8_linear(
            x, weight, weight_scale, None, torch.bfloat16, convrot, 256
        )
        torch.cuda.synchronize()
        delta = (cuda_output.float() - triton_output.float()).abs()
        rows.append(
            {
                "convrot": convrot,
                "bitwise_equal": torch.equal(cuda_output, triton_output),
                "max_abs": float(delta.max().item()),
                "rmse": float(delta.square().mean().sqrt().item()),
                "cuda_max_abs": float(cuda_output.float().abs().max().item()),
                "triton_max_abs": float(triton_output.float().abs().max().item()),
            }
        )
        del cuda_output, triton_output
    return {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "comfy_kitchen_version": "0.2.40",
        "shape": {"M": m, "N": n, "K": k},
        "activation_dtype": "bfloat16",
        "weight_dtype": "int8",
        "output_dtype": "bfloat16",
        "seed": seed,
        "results": rows,
        "interpretation": "Synthetic parity smoke only; no performance or model-quality inference.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=512)
    args = parser.parse_args()
    result = run(args.seed, args.m, args.n, args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
