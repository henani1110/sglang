#!/usr/bin/env python3
"""Run one local-only ChartQA calibration arm against an existing E server script."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chartqa_relaxed_scorer_e import exact_match, relaxed_accuracy, year_strict_correct


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def port_closed(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def gpu_snapshot() -> list[dict[str, str]] | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=15)
    except (subprocess.SubprocessError, OSError):
        return None
    return [{"index": line.split(",")[0].strip(), "name": line.split(",")[1].strip(),
             "utilization_gpu": line.split(",")[2].strip(), "memory_used": line.split(",")[3].strip(),
             "memory_total": line.split(",")[4].strip()} for line in result.stdout.splitlines() if len(line.split(",")) == 5]


def start_sampler(path: Path, interval_ms: int) -> subprocess.Popen[bytes] | None:
    if interval_ms <= 0:
        return None
    stream = path.open("wb")
    stream.write(b"timestamp,index,memory.used [MiB],memory.total [MiB]\n")
    stream.flush()
    proc = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,index,memory.used,memory.total", "--format=csv,noheader", "-lms", str(interval_ms)],
        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    setattr(proc, "_output_stream", stream)
    return proc


def stop_sampler(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    getattr(proc, "_output_stream").close()


def stop_server(proc: subprocess.Popen[bytes] | None) -> dict[str, Any] | None:
    if proc is None:
        return None
    result: dict[str, Any] = {"pid": proc.pid, "signals": [], "stopped": False}
    if proc.poll() is not None:
        result["return_code"] = proc.returncode
        result["stopped"] = True
        return result
    try:
        os.killpg(proc.pid, signal.SIGINT)
        result["signals"].append("SIGINT")
    except ProcessLookupError:
        pass
    try:
        result["return_code"] = proc.wait(timeout=60)
        result["stopped"] = True
        return result
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        result["signals"].append("SIGTERM")
    except ProcessLookupError:
        pass
    try:
        result["return_code"] = proc.wait(timeout=30)
        result["stopped"] = True
    except subprocess.TimeoutExpired:
        result["return_code"] = proc.poll()
    return result


def request_one(candidate: dict[str, Any], image_path: Path, edge: int, keep_ratio: float,
                seed: int, max_tokens: int, endpoint: str, run_id: str) -> dict[str, Any]:
    prompt = (
        "Answer the question using the chart. Return only the shortest direct answer, "
        "with no explanation and no ending punctuation.\n\nQuestion: " + candidate["question"])
    body = {
        "model": "qwen3vlpilot",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_path.as_uri()}},
            {"type": "text", "text": prompt},
        ]}],
        "temperature": 0,
        "seed": seed,
        "max_tokens": max_tokens,
        "stop": ["\n"],
        "stream": False,
        "mm_processor_kwargs": {"min_pixels": edge * edge, "max_pixels": edge * edge,
                                "images_kwargs": {"e_keep_ratio": keep_ratio}},
    }
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(endpoint, data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    start_ns = time.monotonic_ns()
    status = None
    error = None
    response_bytes = b""
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            status = response.status
            response_bytes = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        response_bytes = exc.read()
        error = f"HTTPError: {exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        response_bytes = str(exc).encode("utf-8", errors="replace")
    elapsed_ms = (time.monotonic_ns() - start_ns) / 1e6
    try:
        response = json.loads(response_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        response = {}
    choices = response.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    prediction = message.get("content", "") if isinstance(message, dict) else ""
    if not isinstance(prediction, str):
        prediction = ""
    prediction = prediction.strip()
    target = str(candidate["answer"]).strip()
    relaxed = relaxed_accuracy(prediction, target) if prediction else None
    exact = exact_match(prediction, target) if prediction else None
    year_exact = year_strict_correct(prediction, target) if prediction else None
    usage = response.get("usage") or {}
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "sample_id": candidate["sample_id"],
        "quality_split": candidate["quality_split"],
        "source_split": candidate["source_split"],
        "image_id": candidate["image_id"],
        "image_sha256": candidate["image_sha256"],
        "question": candidate["question"],
        "target_answer": target,
        "prediction_raw": (choice.get("message") or {}).get("content", "") if isinstance(choice.get("message"), dict) else "",
        "prediction_for_scoring": prediction,
        "relaxed_accuracy_correct": relaxed,
        "exact_match_correct": exact,
        "year_exact_correct": year_exact,
        "processor_edge": edge,
        "keep_ratio": keep_ratio,
        "arm": "full" if keep_ratio == 1.0 else f"k{round(keep_ratio*100):02d}",
        "seed": seed,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "request_sha256": hashlib.sha256(payload).hexdigest(),
        "scorer": "chartqa-relaxed-accuracy-e-v0.1",
        "http_status": status,
        "elapsed_client_ms": elapsed_ms,
        "error": error,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "response_complete": status == 200 and bool(prediction) and choice.get("finish_reason") == "stop" and error is None,
        "rights_status": candidate["rights_status"],
        "formal_quality_run": False,
        "redistributable": False,
    }


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edge", type=int, choices=(512, 768), required=True)
    p.add_argument("--keep-ratio", type=float, choices=(0.25, 0.5, 0.75, 1.0), required=True)
    p.add_argument("--candidates-jsonl", type=Path, required=True)
    p.add_argument("--images-root", type=Path, required=True)
    p.add_argument("--server-script", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--attempt", default="a1")
    p.add_argument("--port", type=int, default=8285)
    p.add_argument("--seed", type=int, default=20261110)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--start-timeout", type=int, default=900)
    p.add_argument("--memory-sample-interval-ms", type=int, default=250)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    if not 0.0 < args.keep_ratio <= 1.0 or args.max_tokens <= 0:
        raise ValueError("invalid keep ratio or max tokens")
    if not args.attempt.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError("attempt may contain only letters, digits, dot, underscore, or hyphen")
    if not port_closed(args.port):
        raise RuntimeError(f"port {args.port} is already open; refusing to replace another service")
    all_candidates = load_jsonl(args.candidates_jsonl)
    candidates = [c for c in all_candidates if c["quality_split"] == "calibration"]
    if len(candidates) != 80:
        raise ValueError(f"expected 80 calibration QA, got {len(candidates)}")
    root = args.images_root.resolve()
    for candidate in candidates:
        relative = Path(candidate["image_relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe image path: {relative}")
        image_path = (root / relative).resolve()
        image_path.relative_to(root)
        if not image_path.is_file() or sha256_file(image_path) != candidate["image_sha256"]:
            raise ValueError(f"missing or hash-mismatched image: {image_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "predictions.jsonl"
    summary_path = args.output_dir / "manager-summary.json"
    if result_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {result_path}; pass --resume to continue")
    rows_before = load_jsonl(result_path) if result_path.exists() else []
    done = {row["sample_id"] for row in rows_before if row.get("response_complete")}
    todo = [candidate for candidate in candidates if candidate["sample_id"] not in done]
    run_id = f"chartqa-oecd-exploratory-edge{args.edge}-k{round(args.keep_ratio*100):02d}-calibration-{args.attempt}-s20261110"
    max_model_len = 50_000 if args.edge == 512 else 105_000
    kv_bytes = 6_000_000_000 if args.edge == 512 else 12_400_000_000
    env = os.environ.copy()
    env.update({
        "VL_PILOT_PORT": str(args.port),
        "VL_PILOT_LABEL": run_id,
        "VL_PILOT_MAX_MODEL_LEN": str(max_model_len),
        "VL_PILOT_MAX_NUM_SEQS": "1",
        "VL_PILOT_MAX_NUM_BATCHED_TOKENS": "4096",
        "VL_PILOT_KV_CACHE_MEMORY_BYTES": str(kv_bytes),
        "VL_PILOT_MAX_MEDIA_REQUEST_DECODE_BYTES": "1073741824",
        "VL_PILOT_IMAGE_ROOT": str(root),
        "VL_PILOT_PROBE": "0",
        "VL_PILOT_QUEUE_PROBE": "0",
        "VL_IMAGE_KEEP_RATIO": "1.0",
    })
    server_log = args.output_dir / "server-supervisor.log"
    memory_log = args.output_dir / "nvidia-smi-memory.csv"
    summary: dict[str, Any] = {
        "run_id": run_id, "dataset": "ChartQA-H OECD-prefix local exploratory subset",
        "split": "calibration", "selected": len(candidates), "pending_at_start": len(todo),
        "completed_before": len(done), "edge": args.edge, "keep_ratio": args.keep_ratio,
        "arm": "full" if args.keep_ratio == 1.0 else f"k{round(args.keep_ratio*100):02d}", "attempt": args.attempt, "seed": args.seed,
        "max_tokens": args.max_tokens, "max_model_len": max_model_len,
        "kv_cache_memory_bytes": kv_bytes, "candidate_manifest_sha256": sha256_file(args.candidates_jsonl),
        "scorer_sha256": sha256_file(Path(__file__).with_name("chartqa_relaxed_scorer_e.py")),
        "images_root": str(root), "image_preflight_count": len(candidates),
        "formal_quality_run": False, "eligible_quality_items_added": 0,
        "rights_status": "source_license_pending_local_exploratory_evaluation_only",
        "test_annotations_read": False, "flickr_api_or_url_used": False,
        "gpu_memory_before": gpu_snapshot(), "result_jsonl": str(result_path),
        "server_log": str(server_log), "memory_log": str(memory_log), "candidate_errors": [],
    }
    server: subprocess.Popen[bytes] | None = None
    sampler: subprocess.Popen[bytes] | None = None
    server_stream = server_log.open("wb")
    try:
        server = subprocess.Popen(["bash", str(args.server_script)], stdin=subprocess.DEVNULL,
                                  stdout=server_stream, stderr=subprocess.STDOUT,
                                  env=env, start_new_session=True)
        summary["server_pid"] = server.pid
        url = f"http://127.0.0.1:{args.port}/v1"
        deadline = time.monotonic() + args.start_timeout
        while time.monotonic() < deadline:
            if ready(url + "/models"):
                summary["server_ready"] = True
                break
            if server.poll() is not None:
                summary["server_ready"] = False
                summary["server_return_code_before_requests"] = server.returncode
                break
            time.sleep(2)
        else:
            summary["server_ready"] = False
            summary["server_start_timeout"] = True
        if summary.get("server_ready"):
            sampler = start_sampler(memory_log, args.memory_sample_interval_ms)
            with result_path.open("a", encoding="utf-8", newline="\n") as output:
                for candidate in todo:
                    image_path = (root / candidate["image_relative_path"]).resolve()
                    try:
                        row = request_one(candidate, image_path, args.edge, args.keep_ratio,
                                          args.seed, args.max_tokens, url + "/chat/completions", run_id)
                    except Exception as exc:
                        row = {"created_utc": datetime.now(timezone.utc).isoformat(),
                               "run_id": run_id, "sample_id": candidate["sample_id"],
                               "http_status": None, "response_complete": False,
                               "error": f"{type(exc).__name__}: {exc}",
                               "rights_status": candidate["rights_status"],
                               "formal_quality_run": False}
                        summary["candidate_errors"].append(row)
                    output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    output.flush()
                    os.fsync(output.fileno())
                    summary["requests_written"] = int(summary.get("requests_written", 0)) + 1
                    atomic_json(summary_path, summary)
        else:
            summary["requests_written"] = 0
    finally:
        summary["memory_sampler_stop"] = stop_sampler(sampler)
        summary["server_stop"] = stop_server(server)
        server_stream.close()
        summary["gpu_memory_after"] = gpu_snapshot()
        summary["port_closed_after"] = port_closed(args.port)
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.get("server_ready") and summary.get("port_closed_after") else 2


if __name__ == "__main__":
    raise SystemExit(main())
