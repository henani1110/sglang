#!/usr/bin/env python3
"""Paired x16 experiment for exact FC2 LoRA plus indexed residual-gate fusion."""

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
    "minimax-h3-sm89-kernel-20261030/results/x16-fc2-exact-gate-fusion-20261031"
)
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
    "B": {
        "name": "sage2_exact_fc1_fusion",
        "experimental_exact_lora_gate_fusion": False,
    },
    "G": {
        "name": "sage2_exact_fc1_and_fc2_gate_fusion",
        "experimental_exact_lora_gate_fusion": True,
    },
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


def make_payload(prompt_name, seed, key, directory):
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
        "exact_lora_epilogue": True,
        "experimental_exact_lora_silu_fusion": True,
        "experimental_exact_lora_gate_fusion": config[
            "experimental_exact_lora_gate_fusion"
        ],
        "profile": False,
        "num_profiled_timesteps": 0,
        "profile_all_stages": False,
        "perf_dump_path": str(directory / "perf.json"),
    }


def run_one(label, prompt_name, seed, key):
    directory = ROOT / label
    if directory.exists():
        raise RuntimeError(f"refusing to overwrite result directory: {directory}")
    directory.mkdir(parents=True, exist_ok=False)
    payload = make_payload(prompt_name, seed, key, directory)
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

    (directory / "api_poll_wall_seconds.txt").write_text(
        f"{time.monotonic()-started:.6f}\n"
    )
    if job.get("status") != "completed":
        raise RuntimeError(f"{label} failed: {job.get('error')}")
    file_paths = job.get("file_paths", [])
    if len(file_paths) != 1:
        raise RuntimeError(f"{label} expected one output, got {file_paths}")

    source = pathlib.Path(file_paths[0])
    target = directory / source.name
    subprocess.run(["cp", "--", str(source), str(target)], check=True)
    (directory / "sha256.txt").write_text(
        f"{sha256_file(target)}  {target.name}\n"
    )
    (directory / "decoded-stream-hashes.json").write_text(
        json.dumps(decoded_hashes(target), indent=2) + "\n"
    )
    print(f"{label}: complete", flush=True)


def warmups():
    for prompt_name in ("static", "speech"):
        # Candidate first on the static prompt so the one-shot hook captures
        # real FC2 activations before any formal samples are timed.
        order = ("B", "G") if prompt_name == "static" else ("G", "B")
        for key in order:
            run_one(f"warmup-{prompt_name}-{key}", prompt_name, 1201, key)


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
        "peak_reserved_mb": perf["memory_checkpoints"]["runtime_peak"][
            "peak_reserved_mb"
        ],
        "video_yuv_sha256": hashes["video_yuv420p"]["sha256"],
        "audio_pcm_sha256": hashes["audio_pcm_s16le"]["sha256"],
    }


def formal():
    for index, (prompt_name, seed) in enumerate(CASES):
        order = ("B", "G") if index % 2 == 0 else ("G", "B")
        for key in order:
            stem = f"{prompt_name}-seed{seed}"
            run_one(f"{stem}-{key}", prompt_name, seed, key)


def summarize():
    rows = []
    lookup = {}
    for prompt_name, seed in CASES:
        stem = f"{prompt_name}-seed{seed}"
        for key in CONFIGS:
            row = load_row(f"{stem}-{key}", key, prompt_name, seed)
            rows.append(row)
            lookup[(stem, key)] = row

    pairs = []
    for prompt_name, seed in CASES:
        stem = f"{prompt_name}-seed{seed}"
        baseline = lookup[(stem, "B")]
        candidate = lookup[(stem, "G")]
        pairs.append(
            {
                "sample": stem,
                "baseline_ms": baseline["total_duration_ms"],
                "candidate_ms": candidate["total_duration_ms"],
                "absolute_reduction_ms": (
                    baseline["total_duration_ms"] - candidate["total_duration_ms"]
                ),
                "relative_reduction_pct": 100.0 * (
                    1.0 - candidate["total_duration_ms"]
                    / baseline["total_duration_ms"]
                ),
                "video_hash_match": (
                    baseline["video_yuv_sha256"] == candidate["video_yuv_sha256"]
                ),
                "audio_hash_match": (
                    baseline["audio_pcm_sha256"] == candidate["audio_pcm_sha256"]
                ),
            }
        )
    config_stats = {}
    for key in CONFIGS:
        group = [row for row in rows if row["config"] == key]
        config_stats[key] = {
            "description": CONFIGS[key]["name"],
            "n": len(group),
            "mean_ms": statistics.mean(row["total_duration_ms"] for row in group),
            "median_ms": statistics.median(row["total_duration_ms"] for row in group),
            "mean_denoise_ms": statistics.mean(
                row["denoise_duration_ms"] for row in group
            ),
            "mean_decode_ms": statistics.mean(
                row["decode_duration_ms"] for row in group
            ),
            "rows": group,
        }
    summary = {
        "gpu": "GPU0 RTX 4090 PCIe x16",
        "method": "paired B/G candidate attribution; profiler disabled for all formal samples",
        "workload": {
            "prompts": ["static", "speech"],
            "seeds": [1201, 1202, 1203],
            "steps": 8,
            "resolution": "1344x768",
            "duration_seconds": 5,
            "quality": "exact",
        },
        "formal_samples_per_config": 6,
        "configs": config_stats,
        "paired_median_reduction_pct": statistics.median(
            pair["relative_reduction_pct"] for pair in pairs
        ),
        "paired_mean_reduction_pct": statistics.mean(
            pair["relative_reduction_pct"] for pair in pairs
        ),
        "paired_median_absolute_reduction_ms": statistics.median(
            pair["absolute_reduction_ms"] for pair in pairs
        ),
        "all_six_video_hashes_match": all(pair["video_hash_match"] for pair in pairs),
        "all_six_audio_hashes_match": all(pair["audio_hash_match"] for pair in pairs),
        "all_six_pairs_improved": all(
            pair["absolute_reduction_ms"] > 0 for pair in pairs
        ),
        "pairs": pairs,
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key not in ("configs", "pairs")},
            indent=2,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", choices=("warmups", "formal", "summary"))
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.suite == "warmups":
        warmups()
    elif args.suite == "formal":
        formal()
    else:
        summarize()


if __name__ == "__main__":
    main()
