"""Measure Sage1 or Sage2 H3 attention wrapper at the RTX 4090 production shape."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

N, LIVE, HEADS, DIM = 37760, 37743, 56, 128
DTYPE = torch.bfloat16


def wrapper(qkv: torch.Tensor, backend: str) -> torch.Tensor:
    q, k, v = qkv.unbind(dim=1)
    q = q.contiguous()[:LIVE].unsqueeze(0)
    k = k.contiguous()[:LIVE].unsqueeze(0)
    v = v.contiguous()[:LIVE].unsqueeze(0)
    from sageattention import sageattn

    out_live = sageattn(
        q, k, v, tensor_layout="NHD", is_causal=False, sm_scale=DIM**-0.5
    )[0]
    out = torch.zeros((N, HEADS, DIM), device=qkv.device, dtype=DTYPE)
    out[:LIVE].copy_(out_live)
    return out


def run(args: argparse.Namespace) -> dict:
    torch.set_grad_enabled(False)
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    qkv = torch.randn((N, 3, HEADS, DIM), device="cuda", dtype=DTYPE)
    baseline = None
    if args.baseline_output:
        baseline = torch.load(args.baseline_output, map_location="cuda", weights_only=True)

    output = wrapper(qkv, args.backend)
    torch.cuda.synchronize()
    if tuple(output.shape) != (N, HEADS, DIM) or output.dtype != DTYPE:
        raise AssertionError((output.shape, output.dtype))
    finite = bool(torch.isfinite(output).all().item())
    tail_zero = bool(torch.count_nonzero(output[LIVE:]).item() == 0)
    if not finite or not tail_zero:
        raise AssertionError(f"finite={finite}, tail_zero={tail_zero}")
    if args.save_output:
        args.save_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output.cpu(), args.save_output)

    quality = None
    if baseline is not None:
        max_abs = 0.0
        sqerr = 0.0
        count = 0
        exact = True
        for row in range(0, N, 256):
            a = output[row : row + 256].float()
            b = baseline[row : row + 256].float()
            d = (a - b).abs()
            max_abs = max(max_abs, float(d.max().item()))
            sqerr += float(d.square().sum().item())
            count += d.numel()
            exact = exact and torch.equal(output[row : row + 256], baseline[row : row + 256])
        quality = {"bitwise_equal": exact, "max_abs": max_abs, "rmse": (sqerr / count) ** 0.5}

    del output
    for _ in range(args.warmups):
        output = wrapper(qkv, args.backend)
        del output
    torch.cuda.synchronize()
    rounds = []
    for _ in range(args.rounds):
        values = []
        for _ in range(args.repeats):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            output = wrapper(qkv, args.backend)
            stop.record()
            stop.synchronize()
            values.append(float(start.elapsed_time(stop)))
            del output
        rounds.append(statistics.median(values))
    torch.cuda.synchronize()
    payload = {
        "backend": args.backend,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "shape": [N, HEADS, DIM],
        "live": LIVE,
        "padded": N,
        "dtype": str(DTYPE),
        "warmups": args.warmups,
        "repeats_per_round": args.repeats,
        "rounds": args.rounds,
        "round_medians_ms": rounds,
        "pooled_median_ms": statistics.median(rounds),
        "quality_vs_sage1": quality,
        "finite": finite,
        "tail_zero": tail_zero,
        "seed": args.seed,
        "timestamp_unix": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)
    return payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("sage1", "sage2"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline-output", type=Path)
    p.add_argument("--save-output", type=Path)
    p.add_argument("--seed", type=int, default=20261030)
    p.add_argument("--warmups", type=int, default=5)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--rounds", type=int, default=3)
    run(p.parse_args())


if __name__ == "__main__":
    main()
