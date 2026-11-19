#!/usr/bin/env python3
"""Manage one synthetic live text/image replay through E's fail-safe controller."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def gpu_snapshot() -> list[dict[str, str]] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=15,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    rows = []
    for line in out.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 5:
            rows.append(dict(zip(("index", "name", "utilization_gpu", "memory_used", "memory_total"), fields)))
    return rows


def healthy(base_url: str) -> bool:
    try:
        with urlopen(base_url.rstrip("/") + "/health", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def port_closed(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--ssd", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--server-script", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8285")
    parser.add_argument("--port", type=int, default=8285)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--edge", type=int, choices=(512, 768), required=True)
    parser.add_argument("--text-lanes", type=int, choices=(2, 6), required=True)
    parser.add_argument("--arrival-pattern", choices=("uniform", "burst"), required=True)
    parser.add_argument("--image-count", type=int, default=4)
    parser.add_argument("--text-max-tokens", type=int, default=128)
    parser.add_argument("--image-max-tokens", type=int, default=16)
    parser.add_argument("--health-timeout-seconds", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not port_closed(args.port):
        raise SystemExit(f"refusing to start: port {args.port} is already open")
    replay = args.ssd / "scripts/replay_live_joint_controller_e_seeded_v0.2.py"
    controller = args.ssd / "scripts/e_live_joint_controller_e.py"
    policy = args.ssd / "scripts/e_content_budget_policy.py"
    missing = [str(path) for path in (replay, controller, policy, args.server_script) if not path.is_file()]
    if not args.model.is_dir():
        missing.append(str(args.model))
    if missing:
        raise SystemExit("one or more fixed replay/model/server inputs are missing: " + ", ".join(missing))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    asset_dir = args.output_dir / "assets"
    output_json = args.output_dir / "replay-result.json"
    replay_stdout = args.output_dir / "replay.stdout.log"
    server_log = args.output_dir / "server.log"
    label = f"joint-keepall-{args.edge}-{args.text_lanes}lane-{args.arrival_pattern}-s{args.seed}"
    command = [
        str(args.python), str(replay),
        "--model", str(args.model), "--base-url", args.base_url,
        "--output-json", str(output_json), "--asset-dir", str(asset_dir),
        "--text-lanes", str(args.text_lanes), "--image-size", str(args.edge),
        "--text-max-tokens", str(args.text_max_tokens), "--image-count", str(args.image_count),
        "--arrival-pattern", args.arrival_pattern,
        "--image-max-tokens", str(args.image_max_tokens), "--seed", str(args.seed),
    ]
    summary: dict[str, object] = {
        "run_label": label,
        "formal_run": False,
        "arm": "controller admission replay with uncalibrated fail-safe keep-all",
        "scheduler_mode": "N",
        "edge": args.edge,
        "text_lanes": args.text_lanes,
        "arrival_pattern": args.arrival_pattern,
        "image_count": args.image_count,
        "text_max_tokens": args.text_max_tokens,
        "image_max_tokens": args.image_max_tokens,
        "seed": args.seed,
        "model": str(args.model),
        "server_script": str(args.server_script),
        "server_script_sha256": sha256_file(args.server_script),
        "replay_sha256": sha256_file(replay),
        "controller_sha256": sha256_file(controller),
        "content_policy_sha256": sha256_file(policy),
        "gpu_memory_before": gpu_snapshot(),
        "command": command,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    env = os.environ.copy()
    env.update({
        "VL_PILOT_LABEL": label,
        "VL_STAGE_SPLIT_MODE": "N",
        "VL_PILOT_PROBE": "1",
        "VL_PILOT_QUEUE_PROBE": "1",
    })
    server = None
    server_returncode = None
    replay_returncode = None
    try:
        with server_log.open("wb") as stream:
            server = subprocess.Popen(
                ["bash", str(args.server_script)], cwd=str(args.project), env=env,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            summary["server_pid"] = server.pid
            deadline = time.monotonic() + args.health_timeout_seconds
            while time.monotonic() < deadline and server.poll() is None and not healthy(args.base_url):
                time.sleep(2)
            if server.poll() is not None or not healthy(args.base_url):
                raise RuntimeError(f"server failed readiness; returncode={server.poll()}")
            summary["server_ready"] = True
            with replay_stdout.open("wb") as replay_log:
                proc = subprocess.run(command, cwd=str(args.project), env=env,
                                      stdout=replay_log, stderr=subprocess.STDOUT, check=False)
            replay_returncode = proc.returncode
            summary["replay_returncode"] = replay_returncode
            summary["replay_output_json"] = str(output_json)
            if output_json.is_file():
                summary["replay_result_sha256"] = sha256_file(output_json)
    except Exception as exc:
        summary["manager_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if server is not None and server.poll() is None:
            try:
                os.killpg(server.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                server_returncode = server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(server.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    server_returncode = server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server_returncode = server.poll()
        elif server is not None:
            server_returncode = server.returncode
        summary["server_returncode"] = server_returncode
        summary["port_closed_after"] = port_closed(args.port)
        summary["gpu_memory_after"] = gpu_snapshot()
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output_dir / "manager-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if replay_returncode == 0 and summary.get("port_closed_after") else 1


if __name__ == "__main__":
    raise SystemExit(main())
