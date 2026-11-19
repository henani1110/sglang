import hashlib
import json
import pathlib
import statistics
import subprocess
import time
import urllib.request


ROOT = pathlib.Path(
    "/home/chx/projects/sglang-h3-exact-lora-20261029/experiments/"
    "minimax-h3-sm89-kernel-20261030/results/"
    "fc1-lora-silu-fusion-e2e-gpu1"
)
MANIFEST = pathlib.Path(
    "/home/chx/projects/sglang-h3-exact-lora-20261029/experiments/"
    "minimax-h3-sm89-kernel-20261030/results/"
    "fc1-lora-silu-fusion-e2e-gpu1-manifest.json"
)
BASE = "http://127.0.0.1:30014"
PROMPT = (
    "A locked-off macro shot of a mechanical wristwatch beside a glass of water "
    "on a wooden desk; soft daylight, fine metal engraving and stable reflections, "
    "faint room ambience."
)


def api(path, data=None):
    body = None if data is None else json.dumps(data).encode()
    request = urllib.request.Request(
        BASE + path,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


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


def run_one(spec):
    label = spec["label"]
    directory = ROOT / label
    directory.mkdir(parents=True, exist_ok=False)
    payload = {
        "prompt": PROMPT,
        "seed": 1201,
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
        "exact_lora_epilogue": spec["exact_lora_epilogue"],
        "experimental_exact_lora_silu_fusion": spec[
            "experimental_exact_lora_silu_fusion"
        ],
        "profile": False,
        "num_profiled_timesteps": 0,
        "profile_all_stages": False,
        "perf_dump_path": str(directory / "perf.json"),
    }
    (directory / "request.json").write_text(json.dumps(payload, indent=2) + "\n")
    jobs = api("/v1/videos").get("data", [])
    if any(job.get("status") not in ("completed", "failed") for job in jobs):
        raise RuntimeError("dedicated service already has an active request")

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
    if not file_paths:
        raise RuntimeError(f"{label} completed without output media")
    for source in file_paths:
        source = pathlib.Path(source)
        target = directory / source.name
        subprocess.run(["cp", "--", str(source), str(target)], check=True)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        (directory / "sha256.txt").write_text(f"{digest}  {target.name}\n")
        (directory / "decoded-stream-hashes.json").write_text(
            json.dumps(decoded_hashes(target), indent=2) + "\n"
        )
    print(f"{label}: complete", flush=True)


def summarize(specs):
    rows = []
    for spec in specs:
        directory = ROOT / spec["label"]
        perf = json.loads((directory / "perf.json").read_text())
        hashes = json.loads((directory / "decoded-stream-hashes.json").read_text())
        rows.append(
            {
                "label": spec["label"],
                "fusion": spec["experimental_exact_lora_silu_fusion"],
                "total_duration_ms": perf["total_duration_ms"],
                "denoise_duration_ms": next(
                    item["duration_ms"]
                    for item in perf["steps"]
                    if item["name"] == "MiniMaxH3DenoisingStage"
                ),
                "decode_duration_ms": next(
                    item["duration_ms"]
                    for item in perf["steps"]
                    if item["name"] == "MiniMaxH3DecodingStage"
                ),
                "video_yuv_sha256": hashes["video_yuv420p"]["sha256"],
                "audio_pcm_sha256": hashes["audio_pcm_s16le"]["sha256"],
                "peak_reserved_mb": perf["memory_checkpoints"]["runtime_peak"][
                    "peak_reserved_mb"
                ],
            }
        )
    formal = [row for row in rows if row["label"].startswith("abba")]
    baseline = [
        row["total_duration_ms"] for row in formal if not row["fusion"]
    ]
    candidate = [row["total_duration_ms"] for row in formal if row["fusion"]]
    baseline_mean = statistics.mean(baseline)
    candidate_mean = statistics.mean(candidate)
    hashes_match = len(
        {(row["video_yuv_sha256"], row["audio_pcm_sha256"]) for row in formal}
    ) == 1
    result = {
        "formal_baseline_n": len(baseline),
        "formal_candidate_n": len(candidate),
        "baseline_mean_ms": baseline_mean,
        "candidate_mean_ms": candidate_mean,
        "e2e_reduction_pct": (1.0 - candidate_mean / baseline_mean) * 100.0,
        "all_formal_decoded_media_hashes_match": hashes_match,
        "rows": rows,
    }
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    specs = json.loads(MANIFEST.read_text())
    for spec in specs:
        run_one(spec)
    summarize(specs)


if __name__ == "__main__":
    main()
