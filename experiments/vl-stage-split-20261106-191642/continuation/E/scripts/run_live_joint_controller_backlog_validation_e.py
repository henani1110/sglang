#!/usr/bin/env python3
"""Manage one synthetic live text/image replay through E's fail-safe controller."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


class BacklogSnapshotBroker:
    """Receive nonblocking scheduler UDP snapshots and publish a throttled latest-state file."""

    def __init__(self, latest_path: Path, write_interval_ms: float = 10.0):
        self.latest_path = latest_path
        self.latest_path.parent.mkdir(parents=True, exist_ok=True)
        self.write_interval_ns = int(write_interval_ms * 1_000_000)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(0.1)
        self.port = int(self.sock.getsockname()[1])
        self.stop_event = threading.Event()
        self.latest: dict[str, object] | None = None
        self.received = 0
        self.invalid = 0
        self.write_errors = 0
        self.observer_duration_ns: list[int] = []
        self._thread = threading.Thread(
            target=self._run, name="e-backlog-snapshot-broker", daemon=True
        )
        self._thread.start()

    def _persist_latest(self) -> None:
        if self.latest is None:
            return
        temporary = self.latest_path.with_suffix(self.latest_path.suffix + ".tmp")
        try:
            temporary.write_text(
                json.dumps(self.latest, ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.latest_path)
        except OSError:
            self.write_errors += 1

    def _run(self) -> None:
        last_write_ns = 0
        while not self.stop_event.is_set():
            try:
                packet, _ = self.sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            received_ns = time.monotonic_ns()
            try:
                snapshot = json.loads(packet.decode("utf-8"))
                if not isinstance(snapshot, dict) or snapshot.get("kind") != "scheduler_backlog_snapshot":
                    raise ValueError("unexpected snapshot packet")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                self.invalid += 1
                continue
            self.received += 1
            duration_ns = int(snapshot.get("previous_snapshot_observer_total_ns", 0))
            if duration_ns > 0:
                self.observer_duration_ns.append(duration_ns)
            self.latest = {
                "schema_version": 1,
                "broker_received_monotonic_ns": received_ns,
                "snapshot": snapshot,
            }
            if received_ns - last_write_ns >= self.write_interval_ns:
                self._persist_latest()
                last_write_ns = received_ns
        self._persist_latest()

    def close(self) -> None:
        self.stop_event.set()
        try:
            self.sock.close()
        except OSError:
            pass
        self._thread.join(timeout=1.0)

    def summary(self) -> dict[str, object]:
        ordered_duration_ns = sorted(self.observer_duration_ns)
        duration_summary = {
            "count": len(ordered_duration_ns),
            "median_ns": (
                float(statistics.median(ordered_duration_ns))
                if ordered_duration_ns else None
            ),
            "p95_nearest_rank_ns": (
                int(ordered_duration_ns[max(0, (95 * len(ordered_duration_ns) + 99) // 100 - 1)])
                if ordered_duration_ns else None
            ),
            "max_ns": max(ordered_duration_ns) if ordered_duration_ns else None,
        }
        return {
            "port": self.port,
            "latest_path": str(self.latest_path),
            "received": self.received,
            "invalid": self.invalid,
            "write_errors": self.write_errors,
            "previous_snapshot_observer_total_duration_ns": duration_summary,
            "latest_scheduler_step": (
                self.latest.get("snapshot", {}).get("scheduler_step")
                if self.latest is not None else None
            ),
        }


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
    parser.add_argument("--replay-script", default="scripts/replay_live_joint_controller_e_seeded_v0.3.py")
    parser.add_argument("--image-backlog-cost-model", type=Path)
    parser.add_argument("--backlog-snapshot", choices=("on", "off"), default="on")
    args = parser.parse_args()

    if not port_closed(args.port):
        raise SystemExit(f"refusing to start: port {args.port} is already open")
    replay = args.ssd / args.replay_script
    controller = args.ssd / "scripts/e_live_joint_controller_e.py"
    policy = args.ssd / "scripts/e_content_budget_policy.py"
    required_paths = [replay, controller, policy, args.server_script]
    if args.image_backlog_cost_model is not None:
        required_paths.append(args.image_backlog_cost_model)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if not args.model.is_dir():
        missing.append(str(args.model))
    if missing:
        raise SystemExit("one or more fixed replay/model/server inputs are missing: " + ", ".join(missing))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    asset_dir = args.output_dir / "assets"
    output_json = args.output_dir / "replay-result.json"
    replay_stdout = args.output_dir / "replay.stdout.log"
    server_log = args.output_dir / "server.log"
    backlog_snapshot_file = args.output_dir / "scheduler-backlog-latest.json"
    backlog_broker = BacklogSnapshotBroker(backlog_snapshot_file)
    label = (
        f"joint-keepall-{args.edge}-{args.text_lanes}lane-{args.arrival_pattern}"
        f"-s{args.seed}-backlog-{args.backlog_snapshot}"
    )
    command = [
        str(args.python), str(replay),
        "--model", str(args.model), "--base-url", args.base_url,
        "--output-json", str(output_json), "--asset-dir", str(asset_dir),
        "--text-lanes", str(args.text_lanes), "--image-size", str(args.edge),
        "--text-max-tokens", str(args.text_max_tokens), "--image-count", str(args.image_count),
        "--arrival-pattern", args.arrival_pattern,
        "--image-max-tokens", str(args.image_max_tokens), "--seed", str(args.seed),
        "--scheduler-backlog-snapshot-file", str(backlog_snapshot_file),
    ]
    if args.image_backlog_cost_model is not None:
        command.extend(["--image-backlog-cost-model", str(args.image_backlog_cost_model)])
    summary: dict[str, object] = {
        "run_label": label,
        "formal_run": False,
        "arm": "backlog-cost-model diagnostic replay with fail-safe keep-all",
        "image_backlog_cost_model": (
            str(args.image_backlog_cost_model)
            if args.image_backlog_cost_model is not None else None
        ),
        "image_backlog_cost_model_sha256": (
            sha256_file(args.image_backlog_cost_model)
            if args.image_backlog_cost_model is not None else None
        ),
        "replay_script": str(replay),
        "scheduler_mode": "N",
        "edge": args.edge,
        "text_lanes": args.text_lanes,
        "arrival_pattern": args.arrival_pattern,
        "backlog_snapshot_enabled": args.backlog_snapshot == "on",
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
        "VL_PILOT_BACKLOG_SNAPSHOT": "1" if args.backlog_snapshot == "on" else "0",
        "VL_PILOT_BACKLOG_UDP_PORT": str(backlog_broker.port),
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
        backlog_broker.close()
        summary["scheduler_backlog_broker"] = backlog_broker.summary()
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output_dir / "manager-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if replay_returncode == 0 and summary.get("port_closed_after") else 1


if __name__ == "__main__":
    raise SystemExit(main())
