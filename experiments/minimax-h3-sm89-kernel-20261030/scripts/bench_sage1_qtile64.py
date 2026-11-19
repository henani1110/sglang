"""Correctness, resource, and timing checks for Sage1's Q-tile-64 candidate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))
from kernels.sage1_sm89_qtile64 import forward as qtile64_forward
from sageattention import core as sage_core
from sageattention import attn_qk_int8_per_block as sage_attention_module


N = 37760
LIVE = 37743
HEADS = 56
HEAD_DIM = 128
DTYPE = torch.bfloat16
BASELINE_NAME = "baseline_m128_w8_s4"
CANDIDATES = {
    "qtile64_w4_s3": (4, 3),
    "qtile64_w4_s4": (4, 4),
}
BASE_FORWARD = sage_core.attn_false


def set_variant(name: str) -> None:
    if name == BASELINE_NAME:
        sage_core.attn_false = BASE_FORWARD
        return
    warps, stages = CANDIDATES[name]

    def launch(q, k, v, q_scale, k_scale, tensor_layout="HND", output_dtype=torch.float16):
        return qtile64_forward(
            q,
            k,
            v,
            q_scale,
            k_scale,
            tensor_layout=tensor_layout,
            output_dtype=output_dtype,
            num_warps=warps,
            num_stages=stages,
        )

    sage_core.attn_false = launch


def make_pack(length: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn((length, 3, HEADS, HEAD_DIM), device="cuda", dtype=DTYPE)


def full_wrapper(qkv: torch.Tensor, live: int, output_length: int) -> torch.Tensor:
    """Mirror H3's contiguous QKV, Sage1, and zero-padded output handling."""
    q, k, v = qkv.unbind(dim=1)
    q = q.contiguous()[:live].unsqueeze(0)
    k = k.contiguous()[:live].unsqueeze(0)
    v = v.contiguous()[:live].unsqueeze(0)
    live_output = sage_core.sageattn(
        q,
        k,
        v,
        tensor_layout="NHD",
        is_causal=False,
        sm_scale=HEAD_DIM**-0.5,
    )[0]
    if output_length == live:
        return live_output
    output = torch.zeros(
        (output_length, HEADS, HEAD_DIM), device=q.device, dtype=live_output.dtype
    )
    output[:live].copy_(live_output)
    return output


def compare_one(qkv: torch.Tensor, live: int, output_length: int, variant: str) -> dict:
    set_variant(BASELINE_NAME)
    baseline = full_wrapper(qkv, live, output_length)
    torch.cuda.synchronize()
    set_variant(variant)
    candidate = full_wrapper(qkv, live, output_length)
    torch.cuda.synchronize()
    if tuple(candidate.shape) != (output_length, HEADS, HEAD_DIM):
        raise AssertionError(f"unexpected candidate shape {tuple(candidate.shape)}")
    if baseline.dtype != DTYPE or candidate.dtype != DTYPE:
        raise AssertionError(f"unexpected dtypes {baseline.dtype}, {candidate.dtype}")
    if not torch.isfinite(baseline).all().item() or not torch.isfinite(candidate).all().item():
        raise AssertionError("non-finite output")
    if torch.count_nonzero(baseline[live:]).item() or torch.count_nonzero(candidate[live:]).item():
        raise AssertionError("padded output tail is not zero")
    delta = (candidate.float() - baseline.float()).abs()
    result = {
        "variant": variant,
        "bitwise_equal": torch.equal(candidate, baseline),
        "max_abs": float(delta.max().item()),
        "rmse": float(delta.square().mean().sqrt().item()),
        "shape": list(candidate.shape),
        "dtype": str(candidate.dtype),
        "finite": True,
        "tail_zero": True,
    }
    del baseline, candidate
    gc.collect()
    return result


def correctness_suite() -> list[dict]:
    results = []
    cases = [("actual_shape", N, LIVE, N, 1101), ("actual_shape", N, LIVE, N, 1102), ("actual_shape", N, LIVE, N, 1103)]
    cases.extend(("tile_edge", length, length, length, 9000 + length) for length in (63, 64, 65, 127, 128, 129, 255, 256, 257))
    for kind, tensor_length, live, output_length, seed in cases:
        qkv = make_pack(tensor_length, seed)
        before_k = qkv[:, 1].clone() if kind == "actual_shape" and seed == 1101 else None
        for variant in CANDIDATES:
            try:
                item = compare_one(qkv, live, output_length, variant)
                item.update({"kind": kind, "length": tensor_length, "seed": seed})
                results.append(item)
            except Exception as exc:
                results.append(
                    {
                        "kind": kind,
                        "length": tensor_length,
                        "seed": seed,
                        "variant": variant,
                        "bitwise_equal": False,
                        "error": repr(exc),
                    }
                )
                gc.collect()
                torch.cuda.empty_cache()
        if before_k is not None:
            unchanged = torch.equal(before_k, qkv[:, 1])
            results.append({"kind": "input_mutation", "checked": "packed_K", "unchanged": unchanged})
            del before_k
        del qkv
        gc.collect()
        torch.cuda.empty_cache()
    return results


def quartiles(values: list[float]) -> dict:
    ordered = sorted(values)
    q1, _, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
    return {"median_ms": statistics.median(ordered), "q1_ms": q1, "q3_ms": q3, "iqr_ms": q3 - q1}


def measure(fn, warmups: int, repeats: int) -> dict:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    cuda_ms = []
    host_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        host_ms.append((time.perf_counter() - wall_start) * 1000.0)
        cuda_ms.append(float(start.elapsed_time(stop)))
    return {"cuda_event": {"samples_ms": cuda_ms, **quartiles(cuda_ms)}, "host_wall": {"samples_ms": host_ms, **quartiles(host_ms)}}


def device_sample() -> dict:
    fields = "uuid,temperature.gpu,power.draw,memory.used,utilization.gpu,clocks.sm,clocks_throttle_reasons.active"
    raw = subprocess.check_output(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], text=True).splitlines()
    return {"gpu0": raw[0].strip(), "gpu1": raw[1].strip(), "timestamp_unix": time.time()}


def memory_sample() -> dict:
    stats = torch.cuda.memory_stats(0)
    return {
        "allocated_bytes": torch.cuda.memory_allocated(0),
        "reserved_bytes": torch.cuda.memory_reserved(0),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "num_alloc_retries": stats.get("num_alloc_retries", 0),
        "num_ooms": stats.get("num_ooms", 0),
    }


def prepare_prequantized(qkv: torch.Tensor):
    q, k, v = qkv.unbind(dim=1)
    q = q.contiguous()[:LIVE].unsqueeze(0)
    k = k.contiguous()[:LIVE].unsqueeze(0)
    v = v.contiguous()[:LIVE].unsqueeze(0).to(torch.float16)
    k -= k.mean(dim=1, keepdim=True)
    q_int8, q_scale, k_int8, k_scale = sage_core.per_block_int8(
        q, k, sm_scale=HEAD_DIM**-0.5, tensor_layout="NHD"
    )
    return q_int8, k_int8, v, q_scale, k_scale


def write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2) + "\n")


def run_correctness(args: argparse.Namespace) -> dict:
    torch.set_grad_enabled(False)
    torch.cuda.set_device(0)
    results = correctness_suite()
    passed = all(item.get("bitwise_equal", True) and item.get("unchanged", True) for item in results)
    payload = {
        "mode": "correctness",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "shape": [N, HEADS, HEAD_DIM],
        "dtype": str(DTYPE),
        "live": LIVE,
        "padded": N,
        "tiles": {"baseline_M": 128, "candidate_M": 64, "N": 64, "Q_quant": 128, "K_quant": 64},
        "results": results,
        "passed": passed,
        "timestamp_unix": time.time(),
        "sageattention_source": str(Path(sage_attention_module.__file__).resolve()),
    }
    write_json(args.output, payload)
    print("RESULT_JSON", json.dumps({"output": str(args.output), "passed": passed, "checks": len(results)}), flush=True)
    return payload


def run_bench(args: argparse.Namespace) -> dict:
    torch.set_grad_enabled(False)
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    qkv = torch.randn((N, 3, HEADS, HEAD_DIM), device="cuda", dtype=DTYPE)
    correctness_path = args.correctness
    correctness = json.loads(correctness_path.read_text())
    valid = {
        name
        for name in CANDIDATES
        if any(item.get("variant") == name for item in correctness["results"])
        and all(
            item.get("finite", False)
            and item.get("tail_zero", False)
            and "error" not in item
            and (args.allow_nonexact_diagnostic or item.get("bitwise_equal", False))
            for item in correctness["results"]
            if item.get("variant") == name
        )
    }
    if (not args.allow_nonexact_diagnostic and not correctness.get("passed")) or valid != set(CANDIDATES):
        raise RuntimeError(
            f"correctness gate failed: pass={correctness.get('passed')} valid={sorted(valid)}"
        )

    variants = [BASELINE_NAME, *sorted(CANDIDATES)]
    raw_results = []
    orders = []
    partial_path = args.output.with_name(args.output.stem + ".partial.json")
    for round_index in range(args.rounds):
        order = variants.copy()
        random.Random(20261030 + round_index).shuffle(order)
        orders.append(order)
        for variant in order:
            set_variant(variant)
            torch.cuda.reset_peak_memory_stats(0)
            start_device = device_sample()
            full = measure(lambda: full_wrapper(qkv, LIVE, N), args.warmups, args.repeats)
            q_int8, k_int8, v, q_scale, k_scale = prepare_prequantized(qkv)
            kernel = measure(
                lambda: sage_core.attn_false(q_int8, k_int8, v, q_scale, k_scale, tensor_layout="NHD", output_dtype=DTYPE),
                args.warmups,
                args.repeats,
            )
            end_device = device_sample()
            raw_results.append(
                {
                    "round": round_index + 1,
                    "variant": variant,
                    "is_baseline": variant == BASELINE_NAME,
                    "full_wrapper": full,
                    "prequantized_attention": kernel,
                    "memory": memory_sample(),
                    "device_start": start_device,
                    "device_end": end_device,
                }
            )
            write_json(
                partial_path,
                {
                    "mode": "bench_partial",
                    "completed_records": len(raw_results),
                    "round_order": orders,
                    "results": raw_results,
                    "timestamp_unix": time.time(),
                },
            )
            del q_int8, k_int8, v, q_scale, k_scale
            gc.collect()
            torch.cuda.empty_cache()

    per_round_full = {
        name: [r["full_wrapper"]["cuda_event"]["median_ms"] for r in raw_results if r["variant"] == name]
        for name in variants
    }
    pooled_full_samples = {
        name: [sample for r in raw_results if r["variant"] == name for sample in r["full_wrapper"]["cuda_event"]["samples_ms"]]
        for name in variants
    }
    baseline_pooled = statistics.median(pooled_full_samples[BASELINE_NAME])
    candidate_summary = {
        name: {
            "per_round_median_ms": per_round_full[name],
            "pooled_median_ms": statistics.median(pooled_full_samples[name]),
            "speedup_percent_vs_baseline": (baseline_pooled / statistics.median(pooled_full_samples[name]) - 1.0) * 100.0,
            "all_three_rounds_faster": all(a < b for a, b in zip(per_round_full[name], per_round_full[BASELINE_NAME])),
        }
        for name in CANDIDATES
    }
    payload = {
        "mode": "bench",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "shape": [N, HEADS, HEAD_DIM],
        "dtype": str(DTYPE),
        "live": LIVE,
        "padded": N,
        "tiles": {"baseline_M": 128, "candidate_M": 64, "N": 64, "Q_quant": 128, "K_quant": 64},
        "warmups_per_round_variant": args.warmups,
        "repeats_per_round_variant": args.repeats,
        "rounds": args.rounds,
        "round_order": orders,
        "correctness_file": str(correctness_path),
        "correctness_passed": correctness.get("passed"),
        "diagnostic_allow_nonexact": args.allow_nonexact_diagnostic,
        "baseline_variant": BASELINE_NAME,
        "candidate_summary": candidate_summary,
        "results": raw_results,
        "timestamp_unix": time.time(),
        "git_head": subprocess.check_output(["git", "-C", str(args.worktree), "rev-parse", "HEAD"], text=True).strip(),
        "git_status_sha256": hashlib.sha256(subprocess.check_output(["git", "-C", str(args.worktree), "diff", "HEAD"])).hexdigest(),
        "sageattention_source": str(Path(sage_attention_module.__file__).resolve()),
    }
    write_json(args.output, payload)
    partial_path.unlink(missing_ok=True)
    print("RESULT_JSON", json.dumps({"output": str(args.output), "summary": candidate_summary}), flush=True)
    return payload


def run_profile(args: argparse.Namespace) -> None:
    torch.set_grad_enabled(False)
    torch.cuda.set_device(0)
    qkv = make_pack(N, 4321)
    variant = args.variant
    set_variant(variant)
    for _ in range(args.profile_warmups):
        output = full_wrapper(qkv, LIVE, N)
        del output
    torch.cuda.synchronize()
    output = full_wrapper(qkv, LIVE, N)
    torch.cuda.synchronize()
    print("PROFILE_LAUNCH_COMPLETE", json.dumps({"variant": variant, "shape": [LIVE, HEADS, HEAD_DIM]}), flush=True)
    del output, qkv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("correctness", "bench", "profile"), required=True)
    parser.add_argument("--output", type=Path, default=Path("results/unused.json"))
    parser.add_argument("--correctness", type=Path)
    parser.add_argument("--worktree", type=Path, default=Path("/home/chx/projects/sglang-h3-exact-lora-20261029"))
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--allow-nonexact-diagnostic",
        action="store_true",
        help="time non-bitwise candidates after finite/tail checks; results are diagnostic only",
    )
    parser.add_argument("--variant", choices=(BASELINE_NAME, *CANDIDATES), default=BASELINE_NAME)
    parser.add_argument("--profile-warmups", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "correctness":
        run_correctness(args)
    elif args.mode == "bench":
        if args.correctness is None:
            parser.error("--correctness is required in bench mode")
        run_bench(args)
    else:
        run_profile(args)


if __name__ == "__main__":
    main()
