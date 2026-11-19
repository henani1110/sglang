#!/usr/bin/env python3
"""Run the x16 five-arm MiniMax-H3 attribution matrix and short profiles."""

import argparse
import hashlib
import json
import pathlib
import statistics
import subprocess
import time
import urllib.request


ROOT = pathlib.Path(
    "/home/chx/projects/sglang-h3-exact-lora-20261029/experiments/"
    "minimax-h3-sm89-kernel-20261030/results/x16-five-arm-attribution-20261031"
)
TRACE_ROOT = ROOT / "traces"
BASE = "http://127.0.0.1:30014"
PROMPTS = {
    "static": (
        "A locked-off macro shot of a mechanical wristwatch beside a glass of water "
        "on a wooden desk; soft daylight, fine metal engraving and stable reflections, "
        "faint room ambience."
    ),
    "speech": (
        'A medium close-up of one person facing the camera, clearly saying '
        '"Good morning, the weather is beautiful today" once, with natural mouth '
        "movement and a quiet background."
    ),
}
CASES = [(prompt, seed) for prompt in ("static", "speech") for seed in (1201, 1202, 1203)]
CONFIGS = {
    "A": {"name": "fa_no_epilogue", "backend": "fa", "exact_lora_epilogue": False, "fusion": False},
    "D": {"name": "fa_exact_epilogue", "backend": "fa", "exact_lora_epilogue": True, "fusion": False},
    "E": {"name": "sage2_no_epilogue", "backend": "sage2", "exact_lora_epilogue": False, "fusion": False},
    "B": {"name": "sage2_exact_epilogue", "backend": "sage2", "exact_lora_epilogue": True, "fusion": False},
    "C": {"name": "sage2_exact_epilogue_fc1_fusion", "backend": "sage2", "exact_lora_epilogue": True, "fusion": True},
}


def api(path, data=None):
    body = None if data is None else json.dumps(data).encode()
    request = urllib.request.Request(
        BASE + path,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def sha256_file(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_hashes(path):
    mappings = {
        "video_yuv420p": [
            "-map", "0:v:0", "-vsync", "0", "-pix_fmt", "yuv420p",
            "-f", "rawvideo", "-",
        ],
        "audio_pcm_s16le": [
            "-map", "0:a:0", "-ac", "2", "-ar", "48000", "-f", "s16le", "-",
        ],
    }
    result = {}
    for name, args in mappings.items():
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), *args],
            check=True,
            capture_output=True,
        ).stdout
        result[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return result


def make_payload(prompt_name, seed, key, directory, profile=False, all_stages=False):
    config = CONFIGS[key]
    return {
        "prompt": PROMPTS[prompt_name],
        "seed": seed,
        "num_inference_steps": 8,
        "flow_shift": 12,
        "quality": "exact",
        "task": "t2va",
        "conditions": [],
        "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0},
        "audio_flow_shift": 3,
        "exact_lora_epilogue": config["exact_lora_epilogue"],
        "experimental_exact_lora_silu_fusion": config["fusion"],
        "profile": profile,
        "num_profiled_timesteps": 1 if profile and not all_stages else 0,
        "profile_all_stages": all_stages,
        "perf_dump_path": str(directory / "perf.json"),
    }


def run_one(label, prompt_name, seed, key, profile=False, all_stages=False):
    directory = ROOT / label
    if directory.exists():
        required = ["status.json", "perf.json", "decoded-stream-hashes.json"]
        if all((directory / item).is_file() for item in required):
            status = json.loads((directory / "status.json").read_text())
            if status.get("status") == "completed":
                print(f"{label}: reusing completed result", flush=True)
                return
        raise RuntimeError(f"refusing to overwrite incomplete result directory: {directory}")
    directory.mkdir(parents=True, exist_ok=False)
    payload = make_payload(prompt_name, seed, key, directory, profile, all_stages)
    (directory / "request.json").write_text(json.dumps(payload, indent=2) + "\n")

    jobs = api("/v1/videos").get("data", [])
    if any(job.get("status") not in ("completed", "failed") for job in jobs):
        raise RuntimeError("service already has an active request")

    started = time.monotonic()
    job = api("/v1/videos", payload)
    (directory / "submitted.json").write_text(json.dumps(job, indent=2) + "\n")
    job_id = job["id"]
    deadline = started + 1200
    last_report = started
    while time.monotonic() < deadline:
        job = api("/v1/videos/" + job_id)
        (directory / "status.json").write_text(json.dumps(job, indent=2) + "\n")
        now = time.monotonic()
        if now - last_report >= 30:
            print(
                f"{label}: {now-started:.0f}s {job.get('status')} "
                f"progress={job.get('progress')}",
                flush=True,
            )
            last_report = now
        if job.get("status") in ("completed", "failed"):
            break
        time.sleep(5)
    else:
        raise TimeoutError(f"{label} exceeded 1200 seconds")

    (directory / "api_poll_wall_seconds.txt").write_text(f"{time.monotonic()-started:.6f}\n")
    if job.get("status") != "completed":
        raise RuntimeError(f"{label} failed: {job.get('error')}")
    file_paths = job.get("file_paths", [])
    if len(file_paths) != 1:
        raise RuntimeError(f"{label} expected one output, got {file_paths}")

    source = pathlib.Path(file_paths[0])
    target = directory / source.name
    subprocess.run(["cp", "--", str(source), str(target)], check=True)
    (directory / "sha256.txt").write_text(f"{sha256_file(target)}  {target.name}\n")
    (directory / "decoded-stream-hashes.json").write_text(
        json.dumps(decoded_hashes(target), indent=2) + "\n"
    )
    if profile:
        traces = sorted(str(path) for path in TRACE_ROOT.glob("*.trace.json.gz"))
        (directory / "trace-files.json").write_text(json.dumps(traces, indent=2) + "\n")
    print(f"{label}: complete", flush=True)


def load_row(label, key, prompt_name, seed):
    directory = ROOT / label
    status = json.loads((directory / "status.json").read_text())
    perf = json.loads((directory / "perf.json").read_text())
    hashes = json.loads((directory / "decoded-stream-hashes.json").read_text())
    stages = {item["name"]: item["duration_ms"] for item in perf["steps"]}
    return {
        "label": label,
        "sample": f"{prompt_name}-seed{seed}",
        "config": key,
        "config_name": CONFIGS[key]["name"],
        "total_duration_ms": perf["total_duration_ms"],
        "api_inference_time_s": status.get("inference_time_s"),
        "denoise_duration_ms": stages["MiniMaxH3DenoisingStage"],
        "decode_duration_ms": stages["MiniMaxH3DecodingStage"],
        "peak_reserved_mb": perf["memory_checkpoints"]["runtime_peak"]["peak_reserved_mb"],
        "video_yuv_sha256": hashes["video_yuv420p"]["sha256"],
        "audio_pcm_sha256": hashes["audio_pcm_s16le"]["sha256"],
    }


def reduction(before, after):
    return 100.0 * (1.0 - after / before)


def summarize():
    rows = []
    for prompt_name, seed in CASES:
        stem = f"{prompt_name}-seed{seed}"
        for key in CONFIGS:
            rows.append(load_row(f"{stem}-{key}", key, prompt_name, seed))
    lookup = {(row["sample"], row["config"]): row for row in rows}

    paired = []
    for prompt_name, seed in CASES:
        sample = f"{prompt_name}-seed{seed}"
        arm = {key: lookup[(sample, key)] for key in CONFIGS}
        paired.append({
            "sample": sample,
            "A_fa_no_epilogue_ms": arm["A"]["total_duration_ms"],
            "D_fa_exact_epilogue_ms": arm["D"]["total_duration_ms"],
            "E_sage2_no_epilogue_ms": arm["E"]["total_duration_ms"],
            "B_sage2_exact_epilogue_ms": arm["B"]["total_duration_ms"],
            "C_sage2_exact_fc1_fusion_ms": arm["C"]["total_duration_ms"],
            "fa_epilogue_gain_pct": reduction(arm["A"]["total_duration_ms"], arm["D"]["total_duration_ms"]),
            "sage2_epilogue_gain_pct": reduction(arm["E"]["total_duration_ms"], arm["B"]["total_duration_ms"]),
            "attention_gain_without_epilogue_pct": reduction(arm["A"]["total_duration_ms"], arm["E"]["total_duration_ms"]),
            "attention_gain_with_epilogue_pct": reduction(arm["D"]["total_duration_ms"], arm["B"]["total_duration_ms"]),
            "fc1_fusion_gain_pct": reduction(arm["B"]["total_duration_ms"], arm["C"]["total_duration_ms"]),
            "epilogue_attention_interaction_ms": (
                arm["B"]["total_duration_ms"] - arm["D"]["total_duration_ms"]
                - arm["E"]["total_duration_ms"] + arm["A"]["total_duration_ms"]
            ),
            "A_D_video_hash_match": arm["A"]["video_yuv_sha256"] == arm["D"]["video_yuv_sha256"],
            "A_D_audio_hash_match": arm["A"]["audio_pcm_sha256"] == arm["D"]["audio_pcm_sha256"],
            "E_B_video_hash_match": arm["E"]["video_yuv_sha256"] == arm["B"]["video_yuv_sha256"],
            "E_B_audio_hash_match": arm["E"]["audio_pcm_sha256"] == arm["B"]["audio_pcm_sha256"],
            "B_C_video_hash_match": arm["B"]["video_yuv_sha256"] == arm["C"]["video_yuv_sha256"],
            "B_C_audio_hash_match": arm["B"]["audio_pcm_sha256"] == arm["C"]["audio_pcm_sha256"],
        })

    reductions = [
        "fa_epilogue_gain_pct", "sage2_epilogue_gain_pct",
        "attention_gain_without_epilogue_pct", "attention_gain_with_epilogue_pct",
        "fc1_fusion_gain_pct",
    ]
    config_stats = {}
    for key in CONFIGS:
        group = [row for row in rows if row["config"] == key]
        config_stats[key] = {
            "description": CONFIGS[key]["name"],
            "n": len(group),
            "mean_ms": statistics.mean(row["total_duration_ms"] for row in group),
            "median_ms": statistics.median(row["total_duration_ms"] for row in group),
            "mean_denoise_ms": statistics.mean(row["denoise_duration_ms"] for row in group),
            "median_denoise_ms": statistics.median(row["denoise_duration_ms"] for row in group),
            "mean_decode_ms": statistics.mean(row["decode_duration_ms"] for row in group),
            "median_decode_ms": statistics.median(row["decode_duration_ms"] for row in group),
            "rows": [row for row in group],
        }
    summary = {
        "gpu": "GPU0 RTX 4090 PCIe x16",
        "method": "five-arm paired attribution; all five arms measured in this run",
        "workload": {"prompts": list(PROMPTS), "seeds": [1201, 1202, 1203], "steps": 8,
                     "resolution": "1344x768", "duration_seconds": 5, "quality": "exact"},
        "formal_samples_per_arm": 6,
        "configs": config_stats,
        "paired_median_reductions_pct": {
            key: statistics.median(pair[key] for pair in paired) for key in reductions
        },
        "median_epilogue_attention_interaction_ms": statistics.median(
            pair["epilogue_attention_interaction_ms"] for pair in paired
        ),
        "correctness": {
            key: all(pair[key] for pair in paired)
            for key in (
                "A_D_video_hash_match", "A_D_audio_hash_match",
                "E_B_video_hash_match", "E_B_audio_hash_match",
                "B_C_video_hash_match", "B_C_audio_hash_match",
            )
        },
        "pairs": paired,
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("configs", "pairs")}, indent=2), flush=True)


def run_fa():
    for prompt_name in ("static", "speech"):
        order = ("A", "D") if prompt_name == "static" else ("D", "A")
        for key in order:
            run_one(f"warmup-{prompt_name}-{key}", prompt_name, 1201, key)
    for index, (prompt_name, seed) in enumerate(CASES):
        order = ("A", "D") if index % 2 == 0 else ("D", "A")
        for key in order:
            run_one(f"{prompt_name}-seed{seed}-{key}", prompt_name, seed, key)


def run_sage2():
    warmup_orders = (("E", "B", "C"), ("C", "B", "E"))
    for prompt_index, prompt_name in enumerate(("static", "speech")):
        for key in warmup_orders[prompt_index]:
            run_one(f"warmup-{prompt_name}-{key}", prompt_name, 1201, key)
    orders = (
        ("E", "B", "C"), ("B", "C", "E"), ("C", "E", "B"),
        ("E", "C", "B"), ("C", "B", "E"), ("B", "E", "C"),
    )
    for index, (prompt_name, seed) in enumerate(CASES):
        for key in orders[index]:
            run_one(f"{prompt_name}-seed{seed}-{key}", prompt_name, seed, key)

    # Fresh, non-timed profile requests after the complete C configuration warmup.
    for phase, profile, all_stages in (
        ("denoise", True, False),
        ("all-stages", True, True),
    ):
        run_one(f"profile-warmup-{phase}", "static", 1201, "C")
        run_one(
            f"profile-{phase}-C-static-seed1201",
            "static", 1201, "C", profile=profile, all_stages=all_stages,
        )
    summarize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", choices=("fa", "sage2", "summary"))
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    TRACE_ROOT.mkdir(parents=True, exist_ok=True)
    if args.suite == "fa":
        run_fa()
    elif args.suite == "sage2":
        run_sage2()
    else:
        summarize()


if __name__ == "__main__":
    main()
