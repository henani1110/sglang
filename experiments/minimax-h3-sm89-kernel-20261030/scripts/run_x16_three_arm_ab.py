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
    "minimax-h3-sm89-kernel-20261030/results/x16-3arm-ab"
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


def api(path, data=None):
    body = None if data is None else json.dumps(data).encode()
    request = urllib.request.Request(
        BASE + path,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def file_sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_hashes(path):
    outputs = {}
    mappings = {
        "video_yuv420p": [
            "-map", "0:v:0", "-vsync", "0", "-pix_fmt", "yuv420p",
            "-f", "rawvideo", "-"
        ],
        "audio_pcm_s16le": [
            "-map", "0:a:0", "-ac", "2", "-ar", "48000", "-f", "s16le", "-"
        ],
    }
    for name, args in mappings.items():
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), *args],
            check=True,
            capture_output=True,
        ).stdout
        outputs[name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    return outputs


def make_payload(prompt_name, seed, config, output_dir):
    return {
        "prompt": PROMPTS[prompt_name],
        "seed": seed,
        "num_inference_steps": 8,
        "flow_shift": 12,
        "quality": "exact",
        "task": "t2va",
        "conditions": [],
        "target": {
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 5.0,
        },
        "audio_flow_shift": 3,
        "exact_lora_epilogue": config["exact_lora_epilogue"],
        "experimental_exact_lora_silu_fusion": config["fusion"],
        "profile": False,
        "num_profiled_timesteps": 0,
        "profile_all_stages": False,
        "perf_dump_path": str(output_dir / "perf.json"),
    }


def run_one(label, prompt_name, seed, config):
    directory = ROOT / label
    directory.mkdir(parents=True, exist_ok=False)
    payload = make_payload(prompt_name, seed, config, directory)
    (directory / "request.json").write_text(json.dumps(payload, indent=2) + "\n")

    jobs = api("/v1/videos").get("data", [])
    if any(job.get("status") not in ("completed", "failed") for job in jobs):
        raise RuntimeError("service already has an active request")

    start = time.monotonic()
    job = api("/v1/videos", payload)
    (directory / "submitted.json").write_text(json.dumps(job, indent=2) + "\n")
    deadline = start + 900
    last_report = start
    while time.monotonic() < deadline:
        job = api("/v1/videos/" + job["id"])
        (directory / "status.json").write_text(json.dumps(job, indent=2) + "\n")
        now = time.monotonic()
        if now - last_report >= 30:
            print(
                f"{label}: {now-start:.0f}s {job.get('status')} "
                f"progress={job.get('progress')}",
                flush=True,
            )
            last_report = now
        if job.get("status") in ("completed", "failed"):
            break
        time.sleep(5)
    else:
        raise TimeoutError(f"{label} exceeded 900 seconds")

    (directory / "api_poll_wall_seconds.txt").write_text(
        f"{time.monotonic()-start:.6f}\n"
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
        f"{file_sha256(target)}  {target.name}\n"
    )
    (directory / "decoded-stream-hashes.json").write_text(
        json.dumps(decoded_hashes(target), indent=2) + "\n"
    )
    print(f"{label}: complete", flush=True)


CONFIGS = {
    "fa": {"name": "sglang_default_fa", "exact_lora_epilogue": False, "fusion": False},
    "sage2_exact": {"name": "sage2_exact_lora", "exact_lora_epilogue": True, "fusion": False},
    "sage2_exact_fused": {"name": "sage2_exact_lora_fc1_fused", "exact_lora_epilogue": True, "fusion": True},
}


def run_fa():
    config = CONFIGS["fa"]
    for prompt_name in ("static", "speech"):
        run_one(
            f"warmup-{prompt_name}-fa",
            prompt_name,
            1201,
            config,
        )
    for prompt_name, seed in CASES:
        run_one(f"{prompt_name}-seed{seed}-fa", prompt_name, seed, config)


def run_sage2():
    b = CONFIGS["sage2_exact"]
    c = CONFIGS["sage2_exact_fused"]
    for prompt_name in ("static", "speech"):
        run_one(f"warmup-{prompt_name}-sage2-exact", prompt_name, 1201, b)
        run_one(f"warmup-{prompt_name}-sage2-fused", prompt_name, 1201, c)

    for index, (prompt_name, seed) in enumerate(CASES):
        order = (("sage2_exact", b), ("sage2_exact_fused", c))
        if index % 2:
            order = tuple(reversed(order))
        for key, config in order:
            suffix = "sage2-exact" if key == "sage2_exact" else "sage2-fused"
            run_one(f"{prompt_name}-seed{seed}-{suffix}", prompt_name, seed, config)

    summarize()


def load_row(label, config_key):
    directory = ROOT / label
    status = json.loads((directory / "status.json").read_text())
    perf = json.loads((directory / "perf.json").read_text())
    hashes = json.loads((directory / "decoded-stream-hashes.json").read_text())
    stages = {item["name"]: item["duration_ms"] for item in perf["steps"]}
    return {
        "label": label,
        "config": config_key,
        "total_duration_ms": perf["total_duration_ms"],
        "inference_time_s": status.get("inference_time_s"),
        "denoise_duration_ms": stages["MiniMaxH3DenoisingStage"],
        "decode_duration_ms": stages["MiniMaxH3DecodingStage"],
        "peak_reserved_mb": perf["memory_checkpoints"]["runtime_peak"]["peak_reserved_mb"],
        "video_yuv_sha256": hashes["video_yuv420p"]["sha256"],
        "audio_pcm_sha256": hashes["audio_pcm_s16le"]["sha256"],
    }


def summarize():
    rows = []
    for prompt_name, seed in CASES:
        stem = f"{prompt_name}-seed{seed}"
        rows.extend(
            [
                load_row(f"{stem}-fa", "fa"),
                load_row(f"{stem}-sage2-exact", "sage2_exact"),
                load_row(f"{stem}-sage2-fused", "sage2_exact_fused"),
            ]
        )

    groups = {
        key: [row for row in rows if row["config"] == key]
        for key in CONFIGS
    }
    pairs = []
    for prompt_name, seed in CASES:
        stem = f"{prompt_name}-seed{seed}"
        by_label = {row["label"]: row for row in rows}
        fa = by_label[f"{stem}-fa"]
        exact = by_label[f"{stem}-sage2-exact"]
        fused = by_label[f"{stem}-sage2-fused"]
        pairs.append(
            {
                "sample": stem,
                "fa_ms": fa["total_duration_ms"],
                "sage2_exact_ms": exact["total_duration_ms"],
                "sage2_fused_ms": fused["total_duration_ms"],
                "fa_to_sage2_exact_reduction_pct": 100.0 * (1.0 - exact["total_duration_ms"] / fa["total_duration_ms"]),
                "fa_to_sage2_fused_reduction_pct": 100.0 * (1.0 - fused["total_duration_ms"] / fa["total_duration_ms"]),
                "exact_to_fused_reduction_pct": 100.0 * (1.0 - fused["total_duration_ms"] / exact["total_duration_ms"]),
                "exact_fused_video_hash_match": exact["video_yuv_sha256"] == fused["video_yuv_sha256"],
                "exact_fused_audio_hash_match": exact["audio_pcm_sha256"] == fused["audio_pcm_sha256"],
                "fa_exact_video_hash_match": fa["video_yuv_sha256"] == exact["video_yuv_sha256"],
                "fa_exact_audio_hash_match": fa["audio_pcm_sha256"] == exact["audio_pcm_sha256"],
            }
        )

    summary = {
        "gpu": "GPU0 RTX 4090 PCIe x16",
        "workload": {"prompts": 2, "seeds": [1201, 1202, 1203], "steps": 8, "resolution": "1344x768", "duration_seconds": 5},
        "configs": {
            key: {
                "description": CONFIGS[key]["name"],
                "n": len(items),
                "mean_ms": statistics.mean(row["total_duration_ms"] for row in items),
                "median_ms": statistics.median(row["total_duration_ms"] for row in items),
                "mean_denoise_ms": statistics.mean(row["denoise_duration_ms"] for row in items),
                "mean_decode_ms": statistics.mean(row["decode_duration_ms"] for row in items),
                "rows": items,
            }
            for key, items in groups.items()
        },
        "paired": {
            "fa_to_sage2_exact_median_reduction_pct": statistics.median(pair["fa_to_sage2_exact_reduction_pct"] for pair in pairs),
            "fa_to_sage2_fused_median_reduction_pct": statistics.median(pair["fa_to_sage2_fused_reduction_pct"] for pair in pairs),
            "exact_to_fused_median_reduction_pct": statistics.median(pair["exact_to_fused_reduction_pct"] for pair in pairs),
            "all_exact_fused_video_hashes_match": all(pair["exact_fused_video_hash_match"] for pair in pairs),
            "all_exact_fused_audio_hashes_match": all(pair["exact_fused_audio_hash_match"] for pair in pairs),
            "pairs": pairs,
        },
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["paired"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", choices=("fa", "sage2"))
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.suite == "fa":
        run_fa()
    else:
        run_sage2()


if __name__ == "__main__":
    main()
