#!/usr/bin/env python3
"""Execute one frozen E online system run on the SSD-backed GPU0 host."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

from PIL import Image


RUN_ID = "vl-stage-split-20261106-191642"
DEFAULT_PROJECT = Path(f"/home/chx/project/{RUN_ID}/continuation/E")
DEFAULT_SSD = Path(f"/media/chx/Win10/{RUN_ID}/continuation/E")
DEFAULT_E_CAMPAIGN = Path(f"/home/chx/project/{RUN_ID}/continuation/E/scripts/campaign_e_online.py")
DEFAULT_PYTHON = Path("/home/chx/project/vl-encode-pilot-20261106-085428/env/serve311/bin/python")
MODEL = Path("/media/chx/Win10/vl-encode-pilot-20261106-085428/models/Qwen3-VL-2B-Instruct")
FROZEN_D_PROFILE_SHA256 = "92bb071b94599533a79113f2bf519dff785b75788ffd27ccd7d8ffe0c9ac9483"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def gpu_snapshot() -> list[dict[str, str]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    rows = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 5:
            rows.append(dict(zip(("index", "name", "utilization_gpu", "memory_used", "memory_total"), values)))
    return rows


def port_closed(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def healthy(url: str) -> bool:
    try:
        with urlopen(url.rstrip("/") + "/health", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def client_phase_durations(events_path: Path, result: dict[str, object]) -> tuple[float | None, float | None]:
    warmup_start_ns: int | None = None
    warmup_drain_seconds: float | None = None
    measurement_start_ns = int(result["measurement_start_ns"]) if result.get("measurement_start_ns") is not None else None
    measurement_end_ns: int | None = None
    with events_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            kind = row.get("kind")
            phase = row.get("phase")
            if kind == "phase_start" and phase == "warmup" and warmup_start_ns is None:
                warmup_start_ns = int(row["time_ns"])
            elif kind == "warmup_drain_complete":
                warmup_drain_seconds = float(row["actual_duration_ms"]) / 1000.0
            elif kind == "phase_end" and phase == "measurement":
                measurement_end_ns = int(row["time_ns"])
    if warmup_drain_seconds is None and warmup_start_ns is not None and measurement_start_ns is not None:
        warmup_drain_seconds = (measurement_start_ns - warmup_start_ns) / 1e9
    measurement_seconds = (
        (measurement_end_ns - measurement_start_ns) / 1e9
        if measurement_end_ns is not None and measurement_start_ns is not None
        else None
    )
    return warmup_drain_seconds, measurement_seconds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--ssd", type=Path, default=DEFAULT_SSD)
    parser.add_argument("--workload-dir", type=Path)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_E_CAMPAIGN)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--arm", choices=("N", "A", "B-fixed", "B-content", "B-joint"), default="N")
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--edge", type=int, choices=(512, 768), default=512)
    parser.add_argument("--text-lanes", type=int, choices=(2, 4, 6), default=2)
    parser.add_argument("--arrival-mode", choices=("uniform", "burst"), default="uniform")
    parser.add_argument("--warmup-seconds", type=int, default=60)
    parser.add_argument("--measure-seconds", type=int, default=180)
    parser.add_argument("--port", type=int, default=8285)
    parser.add_argument("--health-timeout-seconds", type=int, default=300)
    parser.add_argument("--block-id", default="20261111-first-paired-block")
    args = parser.parse_args()

    if not port_closed(args.port):
        raise SystemExit(f"refusing to start: port {args.port} is already listening")
    default_workload = (
        args.ssd / "datasets/online-synthetic-v0.1"
        if args.edge == 512
        else args.ssd / "datasets/online-synthetic-v0.2-g768"
    )
    image_dir = args.workload_dir or default_workload
    image_manifest = image_dir / "manifest.json"
    server_script = args.project / "scripts/run_server_e_deadline_a.sh" if args.arm == "A" else args.project / "scripts/run_server_e.sh"
    if not image_manifest.is_file() or not server_script.is_file() or not args.campaign.is_file() or not MODEL.is_dir():
        raise SystemExit("one or more frozen E workload, server, campaign, or model inputs are missing")
    deadline_profile = args.ssd.parent / "D/manifests/estimator-profile.json"
    deadline_profile_sha256 = None
    if args.arm == "A":
        if not deadline_profile.is_file():
            raise SystemExit(f"frozen D deadline profile is missing: {deadline_profile}")
        deadline_profile_sha256 = sha256(deadline_profile)
        if deadline_profile_sha256 != FROZEN_D_PROFILE_SHA256:
            raise SystemExit(f"frozen D deadline profile hash mismatch: {deadline_profile_sha256}")
    image_data = json.loads(image_manifest.read_text(encoding="utf-8"))
    expected_dataset_id = (
        "e-online-synthetic-charts-v0.1" if args.edge == 512 else "e-online-synthetic-charts-v0.2-g768"
    )
    if (
        image_data.get("dataset_id") != expected_dataset_id
        or int(image_data.get("selected_count", 0)) != 4
        or int(image_data.get("size", 0)) != args.edge
    ):
        raise SystemExit(f"online synthetic image manifest does not match the requested {args.edge}-pixel cell")
    for row in image_data["images"]:
        image_path = image_dir / Path(row["path"]).name
        if sha256(image_path) != row["sha256"]:
            raise SystemExit(f"synthetic image SHA-256 mismatch: {image_path.name}")
        with Image.open(image_path) as image:
            if image.size != (args.edge, args.edge):
                raise SystemExit(
                    f"synthetic image {image_path.name} is {image.width}x{image.height}, expected {args.edge}x{args.edge}"
                )

    attempt = f"a{args.attempt}"
    run_key = f"{args.block_id}/E-{args.arm}-G{args.edge}-L{args.text_lanes}-{args.arrival_mode}-s{args.seed}-{attempt}"
    output_dir = args.ssd / "runs/formal-online-v0.1" / run_key
    log_dir = args.ssd / "logs/formal-online-v0.1" / run_key
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite prior run directory: {output_dir}")
    output_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    result_path = output_dir / "result.json"
    events_path = output_dir / "client-events.jsonl"
    control_path = args.ssd / "configs/pilot_control.json"
    server_log = log_dir / "server.log"
    client_log = log_dir / "client.log"
    base_url = f"http://127.0.0.1:{args.port}"
    run_label = f"e-online-{args.arm}-g{args.edge}-l{args.text_lanes}-{args.arrival_mode}-s{args.seed}-{attempt}-{args.block_id}"
    run_id = f"E-{args.arm}-G{args.edge}-L{args.text_lanes}-{args.arrival_mode}-s{args.seed}-{attempt}"
    client_command = [
        str(args.python), str(args.campaign),
        "--base-url", base_url, "--model-name", "qwen3vlpilot", "--model-path", str(MODEL),
        "--control-image-dir", str(image_dir), "--image-size", str(args.edge),
        "--text-streams", str(args.text_lanes), "--arrival-mode", args.arrival_mode,
        "--policy", args.arm, "--seed", str(args.seed), "--block", "1", "--run-id", run_id,
        "--text-tokens", "192", "--image-tokens", "16",
        "--warmup-seconds", str(args.warmup_seconds), "--measure-seconds", str(args.measure_seconds),
        "--control-file", str(control_path), "--client-events", str(events_path),
        "--results-file", str(result_path), "--calibration-diagnostic",
    ]
    environment = os.environ.copy()
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        environment.pop(key, None)
    environment.update({
        "VL_PILOT_LABEL": run_label,
        "VL_STAGE_SPLIT_MODE": "N",
        "VL_IMAGE_KEEP_RATIO": "0.5" if args.arm == "B-fixed" else "1.0",
        "VL_PILOT_PROBE": "1",
        "VL_PILOT_QUEUE_PROBE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    manifest = {
        "schema_version": 1,
        "status": "running",
        "formal_online_run": True,
        "formal_block_complete": False,
        "run_id": run_id,
        "server_log_label": run_label,
        "block_id": args.block_id,
        "arm": args.arm,
        "arm_definition": {
            "N": "E serving path with native scheduler and full visual-token retention",
            "B-fixed": "E residual-rank fixed-K candidate with 0.5 keep ratio",
            "A": "E-owned launch using the frozen D adaptive deadline scheduler and full visual-token retention",
            "B-content": "E residual-rank candidate routed by lexical query and raw-pixel variance proxies",
            "B-joint": "B-content candidate adjusted by live text gap and observed image-TTFT slack",
        }[args.arm],
        "cell": {"grid": args.edge, "active_text_lanes": args.text_lanes, "arrival_mode": args.arrival_mode},
        "seed": args.seed,
        "warmup_seconds": args.warmup_seconds,
        "measurement_seconds": args.measure_seconds,
        "arrival_spec": "uniform: 1 image every 10 seconds; burst: paired arrivals at 0/100ms per 2 seconds (E-owned campaign schedule)",
        "text_tokens_per_request": 192,
        "image_tokens_per_request": 16,
        "image_set": image_data,
        "workload_dir": str(image_dir),
        "image_manifest_sha256": sha256(image_manifest),
        "campaign_path_read_only": str(args.campaign),
        "campaign_sha256": sha256(args.campaign),
        "server_script": str(server_script),
        "server_script_sha256": sha256(server_script),
        "server_mode": "A" if args.arm == "A" else "N",
        "deadline_profile_path": str(deadline_profile) if args.arm == "A" else None,
        "deadline_profile_sha256": deadline_profile_sha256,
        "visual_keep_ratio": 0.5 if args.arm == "B-fixed" else 1.0,
        "gpu_selection": "run_server_e.sh pins GPU0 UUID; GPU1 excluded",
        "model_path": str(MODEL),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "client_command": client_command,
    }
    (output_dir / "run-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary: dict[str, object] = {
        "run_id": run_id,
        "formal_online_run": True,
        "arm": args.arm,
        "gpu_memory_before": gpu_snapshot(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "client_command": client_command,
    }
    server: subprocess.Popen[bytes] | None = None
    client_rc: int | None = None
    try:
        with server_log.open("wb") as stream:
            server = subprocess.Popen(
                ["bash", str(server_script)], cwd=str(args.project), env=environment,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        summary["server_pid"] = server.pid
        deadline = time.monotonic() + args.health_timeout_seconds
        while time.monotonic() < deadline and server.poll() is None and not healthy(base_url):
            time.sleep(2)
        if server.poll() is not None or not healthy(base_url):
            raise RuntimeError(f"E server did not become healthy; exit={server.poll()}")
        summary["server_ready"] = True
        with client_log.open("wb") as stream:
            client = subprocess.run(client_command, cwd=str(args.project), env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT, check=False)
        client_rc = client.returncode
        summary["client_returncode"] = client_rc
        summary["result_path"] = str(result_path)
        if result_path.is_file():
            summary["result_sha256"] = sha256(result_path)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            summary["result"] = result
            warmup_actual, measurement_actual = client_phase_durations(events_path, result)
            summary["warmup_actual_seconds"] = warmup_actual
            summary["measurement_actual_seconds"] = measurement_actual
            summary["successful_measurement_images"] = int(result.get("measurement_image_responses", 0))
            summary["planned_measurement_images"] = int(result.get("measurement_images_planned", 0))
            summary["measurement_image_errors"] = int(result.get("measurement_image_errors", 0))
            summary["text_errors"] = int(result.get("text_errors", 0))
            summary["experiment_arm_eligible"] = bool(
                result.get("status") == "completed"
                and result.get("measurement_images_planned") == result.get("measurement_image_responses")
                and result.get("measurement_image_errors") == 0
                and result.get("text_errors") == 0
                and warmup_actual is not None
                and warmup_actual >= args.warmup_seconds
                and measurement_actual is not None
                and measurement_actual >= args.measure_seconds
            )
    except Exception as exc:
        summary["manager_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if server is not None and server.poll() is None:
            try:
                os.killpg(server.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                summary["server_returncode"] = server.wait(timeout=90)
            except subprocess.TimeoutExpired:
                summary["sigint_shutdown_timeout"] = True
                try:
                    os.killpg(server.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    summary["server_returncode"] = server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    summary["server_shutdown_timeout"] = True
                    summary["server_returncode"] = server.poll()
        elif server is not None:
            summary["server_returncode"] = server.returncode
        port_deadline = time.monotonic() + 30
        while time.monotonic() < port_deadline and not port_closed(args.port):
            time.sleep(1)
        summary["port_closed_after"] = port_closed(args.port)
        summary["gpu_memory_after"] = gpu_snapshot()
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (output_dir / "manager-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest["status"] = "completed" if summary.get("experiment_arm_eligible") else "failed_or_incomplete"
        manifest["finished_utc"] = summary["finished_utc"]
        manifest["warmup_actual_seconds"] = summary.get("warmup_actual_seconds")
        manifest["measurement_actual_seconds"] = summary.get("measurement_actual_seconds")
        manifest["manager_summary_sha256"] = sha256(output_dir / "manager-summary.json")
        (output_dir / "run-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if client_rc == 0 and summary.get("experiment_arm_eligible") and summary.get("port_closed_after") else 1


if __name__ == "__main__":
    raise SystemExit(main())
