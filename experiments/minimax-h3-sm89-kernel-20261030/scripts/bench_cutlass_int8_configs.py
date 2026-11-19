"""Benchmark pre-instantiated comfy-kitchen CUTLASS INT8 configs on H3 shapes."""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch

from comfy_kitchen.backends.cuda import _C


SHAPES = [
    {"name": "M37760_N21504_K5376", "m": 37760, "n": 21504, "k": 5376},
    {"name": "M37760_N5376_K14336", "m": 37760, "n": 5376, "k": 14336},
    {"name": "M37760_N28672_K5376", "m": 37760, "n": 28672, "k": 5376},
]
BASELINE_CONFIG = 13  # selected by comfy-kitchen 0.2.40 for these H3 shapes
CANDIDATES = {
    "config0_128x256x64_identity": 0,
    "config12_128x128x64_streamk": 12,
    "config2_64x128x64": 2,
}


def make_inputs(shape: dict, seed: int):
    torch.manual_seed(seed)
    m, n, k = shape["m"], shape["n"], shape["k"]
    a = torch.randint(-128, 128, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int8)
    x_scale = torch.rand((m,), device="cuda", dtype=torch.float32)
    w_scale = torch.rand((n,), device="cuda", dtype=torch.float32)
    return a, b, x_scale, w_scale


def launch(a, b, x_scale, w_scale, shape: dict, config: int) -> torch.Tensor:
    out = torch.empty((shape["m"], shape["n"]), device="cuda", dtype=torch.bfloat16)
    ok = _C.cutlass_int8_dequant_config(
        a,
        b,
        x_scale,
        w_scale,
        out,
        2,
        config,
        torch.cuda.current_stream().cuda_stream,
    )
    if not ok:
        raise RuntimeError(f"CUTLASS config {config} did not launch for {shape['name']}")
    return out


def correctness_suite(seed: int) -> list[dict]:
    results = []
    for index, shape in enumerate(SHAPES):
        a, b, x_scale, w_scale = make_inputs(shape, seed + index)
        selected = int(_C.cutlass_int8_selected_config(shape["m"], shape["n"], shape["k"]))
        baseline = launch(a, b, x_scale, w_scale, shape, BASELINE_CONFIG)
        torch.cuda.synchronize()
        for name, config in CANDIDATES.items():
            candidate = launch(a, b, x_scale, w_scale, shape, config)
            torch.cuda.synchronize()
            if not torch.isfinite(candidate).all().item():
                raise AssertionError(f"non-finite output for {name}/{shape['name']}")
            bitwise_equal = torch.equal(candidate, baseline)
            if bitwise_equal:
                max_abs = 0.0
                rmse = 0.0
            else:
                # Avoid materializing multi-gigabyte float32 output copies for
                # the largest M/N shape when a config differs.
                max_abs = 0.0
                squared_error = 0.0
                element_count = 0
                for row_start in range(0, shape["m"], 256):
                    row_end = min(shape["m"], row_start + 256)
                    delta = (
                        candidate[row_start:row_end].float()
                        - baseline[row_start:row_end].float()
                    ).abs()
                    max_abs = max(max_abs, float(delta.max().item()))
                    squared_error += float(delta.square().sum().item())
                    element_count += delta.numel()
                rmse = (squared_error / element_count) ** 0.5
            results.append(
                {
                    "shape": shape["name"],
                    "selected_production_config": selected,
                    "baseline_config": BASELINE_CONFIG,
                    "candidate": name,
                    "candidate_config": config,
                    "bitwise_equal": bitwise_equal,
                    "max_abs": max_abs,
                    "rmse": rmse,
                }
            )
            del candidate
        del baseline, a, b, x_scale, w_scale
        gc.collect()
        torch.cuda.empty_cache()
    return results


def quartiles(values: list[float]) -> dict:
    ordered = sorted(values)
    q1, _, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
    return {
        "median_ms": statistics.median(ordered),
        "q1_ms": q1,
        "q3_ms": q3,
        "iqr_ms": q3 - q1,
    }


def measure(a, b, x_scale, w_scale, shape: dict, config: int, warmups: int, repeats: int) -> dict:
    for _ in range(warmups):
        out = launch(a, b, x_scale, w_scale, shape, config)
        del out
    torch.cuda.synchronize()
    samples = []
    host_samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        out = launch(a, b, x_scale, w_scale, shape, config)
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
        host_samples.append((time.perf_counter() - wall_start) * 1000.0)
        del out
    return {
        "cuda_event": {"samples_ms": samples, **quartiles(samples)},
        "host_wall": {"samples_ms": host_samples, **quartiles(host_samples)},
    }


def device_sample() -> dict:
    fields = "uuid,temperature.gpu,power.draw,memory.used,utilization.gpu,clocks.sm"
    raw = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    return {"gpu0": raw[0].strip(), "gpu1": raw[1].strip(), "timestamp_unix": time.time()}


def write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def run_correctness(args: argparse.Namespace) -> dict:
    torch.set_grad_enabled(False)
    torch.cuda.set_device(0)
    results = correctness_suite(args.seed)
    passed = all(row["bitwise_equal"] for row in results)
    payload = {
        "mode": "correctness",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "comfy_kitchen_version": "0.2.40",
        "input": "synthetic int8 activations/weights with per-row/per-channel float scales",
        "output_dtype": "torch.bfloat16",
        "baseline_config": BASELINE_CONFIG,
        "candidate_configs": CANDIDATES,
        "results": results,
        "passed": passed,
        "timestamp_unix": time.time(),
    }
    write_json(args.output, payload)
    print("RESULT_JSON", json.dumps({"output": str(args.output), "passed": passed, "checks": len(results)}), flush=True)
    return payload


def run_bench(args: argparse.Namespace) -> dict:
    torch.set_grad_enabled(False)
    torch.cuda.set_device(0)
    correctness = json.loads(args.correctness.read_text())
    if not correctness.get("passed"):
        raise RuntimeError("CUTLASS config correctness gate failed")
    records = []
    orders = []
    configs = [("baseline_config13", BASELINE_CONFIG), *CANDIDATES.items()]
    partial = args.output.with_name(args.output.stem + ".partial.json")
    for shape_index, shape in enumerate(SHAPES):
        a, b, x_scale, w_scale = make_inputs(shape, args.seed + shape_index)
        shape_records = []
        for round_index in range(args.rounds):
            order = configs.copy()
            random.Random(20261030 + shape_index * 10 + round_index).shuffle(order)
            orders.append({"shape": shape["name"], "round": round_index + 1, "order": order})
            for name, config in order:
                torch.cuda.reset_peak_memory_stats(0)
                start_device = device_sample()
                measured = measure(a, b, x_scale, w_scale, shape, config, args.warmups, args.repeats)
                end_device = device_sample()
                row = {
                    "shape": shape["name"],
                    "round": round_index + 1,
                    "variant": name,
                    "config": config,
                    "measurement": measured,
                    "max_allocated_bytes": torch.cuda.max_memory_allocated(0),
                    "max_reserved_bytes": torch.cuda.max_memory_reserved(0),
                    "device_start": start_device,
                    "device_end": end_device,
                }
                records.append(row)
                shape_records.append(row)
                write_json(
                    partial,
                    {
                        "mode": "bench_partial",
                        "completed_records": len(records),
                        "round_order": orders,
                        "results": records,
                        "timestamp_unix": time.time(),
                    },
                )
        del a, b, x_scale, w_scale
        gc.collect()
        torch.cuda.empty_cache()

    summary = {}
    for shape in SHAPES:
        rows = [r for r in records if r["shape"] == shape["name"]]
        baseline_samples = [s for r in rows if r["variant"] == "baseline_config13" for s in r["measurement"]["cuda_event"]["samples_ms"]]
        baseline_median = statistics.median(baseline_samples)
        summary[shape["name"]] = {
            "baseline_pooled_median_ms": baseline_median,
            "candidates": {},
        }
        for name in CANDIDATES:
            candidate_samples = [s for r in rows if r["variant"] == name for s in r["measurement"]["cuda_event"]["samples_ms"]]
            round_medians = {
                i: next(r["measurement"]["cuda_event"]["median_ms"] for r in rows if r["variant"] == name and r["round"] == i)
                for i in range(1, args.rounds + 1)
            }
            baseline_round_medians = {
                i: next(r["measurement"]["cuda_event"]["median_ms"] for r in rows if r["variant"] == "baseline_config13" and r["round"] == i)
                for i in range(1, args.rounds + 1)
            }
            candidate_median = statistics.median(candidate_samples)
            summary[shape["name"]]["candidates"][name] = {
                "config": CANDIDATES[name],
                "pooled_median_ms": candidate_median,
                "speedup_percent_vs_config13": (baseline_median / candidate_median - 1.0) * 100.0,
                "per_round_median_ms": [round_medians[i] for i in range(1, args.rounds + 1)],
                "all_rounds_faster": all(round_medians[i] < baseline_round_medians[i] for i in range(1, args.rounds + 1)),
            }
    payload = {
        "mode": "bench",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "comfy_kitchen_version": "0.2.40",
        "input": "synthetic prequantized int8 activations/weights with per-row/per-channel float scales",
        "output_dtype": "torch.bfloat16",
        "baseline_config": BASELINE_CONFIG,
        "candidate_configs": CANDIDATES,
        "warmups_per_round_config": args.warmups,
        "repeats_per_round_config": args.repeats,
        "rounds_per_shape": args.rounds,
        "round_order": orders,
        "correctness_file": str(args.correctness),
        "summary": summary,
        "results": records,
        "timestamp_unix": time.time(),
    }
    write_json(args.output, payload)
    partial.unlink(missing_ok=True)
    print("RESULT_JSON", json.dumps({"output": str(args.output), "summary": summary}), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("correctness", "bench"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--correctness", type=Path)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.mode == "correctness":
        run_correctness(args)
    else:
        if args.correctness is None:
            parser.error("--correctness is required in bench mode")
        run_bench(args)


if __name__ == "__main__":
    main()
