#!/usr/bin/env python3
"""Start each candidate profile once, capture server_info, then stop our own process."""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path


ROOT = Path("/home/chx/projects/tokenflow-repro-paper-candidate")
STAGE = ROOT / "repro/stage3-paper-alignment-v1"
OUT = STAGE / "preflight"
PROFILES = [
    "sglang-fork-plain",
    "sglang-fork-chunked-2048",
    "tokenflow-paper-v1-chunked-2048",
]
PORT = 30000


def gpu1_uuid() -> str:
    return subprocess.check_output(
        ["nvidia-smi", "-i", "1", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()


def assert_gpu1_idle() -> None:
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True
    )
    target = gpu1_uuid()
    active = [line.strip() for line in raw.splitlines() if target in line]
    mem = subprocess.check_output(
        ["nvidia-smi", "-i", "1", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    ).strip()
    if active or int(mem) > 200:
        raise RuntimeError(f"GPU1 is not idle; compute_apps={active}, memory_used_mib={mem}")


def assert_port_free() -> None:
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            raise RuntimeError(f"port {PORT} is occupied; refusing to touch its process")


def get_json(path: str, timeout: float = 5.0):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as response:
        return json.loads(response.read())


def check_health() -> None:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5.0) as response:
        response.read()


def stop_ours(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=15)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if commit != "e754998d6980656707b9628fb5121a4ac4f985df":
        raise RuntimeError(f"unexpected source commit: {commit}")
    scheduler = (ROOT / "python/sglang/srt/managers/scheduler.py").read_text()
    if "if self.chunked_prefill_size <= 0" not in scheduler or "self.chunked_prefill_size = None" not in scheduler:
        raise RuntimeError("scheduler no longer contains the audited chunked-prefill disable path")

    expected = {
        "sglang-fork-plain": {"chunked_prefill_size": -1, "enable_hierarchical_cache": False, "kv_offload_policy": "default"},
        "sglang-fork-chunked-2048": {"chunked_prefill_size": 2048, "enable_hierarchical_cache": False, "kv_offload_policy": "default"},
        "tokenflow-paper-v1-chunked-2048": {"chunked_prefill_size": 2048, "enable_hierarchical_cache": True, "kv_offload_policy": "paper_v1"},
    }
    result_rows = []
    for profile in PROFILES:
        result_path = OUT / f"{profile}-server-info.json"
        log_path = OUT / f"{profile}-server.log"
        if result_path.exists() or log_path.exists():
            raise FileExistsError(f"refusing to overwrite prior preflight: {profile}")
        assert_gpu1_idle()
        assert_port_free()
        log = log_path.open("x")
        started = time.time()
        proc = subprocess.Popen(
            ["bash", str(STAGE / "launch_profile.sh"), profile],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        info = None
        try:
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"{profile} exited early with code {proc.returncode}; inspect {log_path}")
                try:
                    check_health()
                    info = get_json("/server_info")
                    break
                except Exception:
                    time.sleep(2)
            if info is None:
                raise TimeoutError(f"{profile} did not become healthy within 900 seconds")
            checks = {
                **expected[profile],
                "dtype": "bfloat16",
                "tp_size": 1,
                "mem_fraction_static": 0.8,
                "max_running_requests": 128,
                "max_total_num_tokens": 77001,
                "attention_backend": "triton",
                "sampling_backend": "pytorch",
                "enable_deterministic_inference": True,
                "disable_piecewise_cuda_graph": True,
            }
            failed = {key: (info.get(key), value) for key, value in checks.items() if info.get(key) != value}
            if failed:
                raise AssertionError(f"{profile} server_info mismatch: {failed}")
            (OUT / f"{profile}-server-info.json").write_text(json.dumps(info, indent=2))
            result_rows.append({"profile": profile, "pid": proc.pid, "start_unix_s": started, "source_commit": commit, "server_info_checks": checks, "passed": True})
            print(json.dumps(result_rows[-1]), flush=True)
        finally:
            stop_ours(proc)
            log.close()
        assert_port_free()
        assert_gpu1_idle()
    (OUT / "preflight-summary.json").write_text(json.dumps(result_rows, indent=2))


if __name__ == "__main__":
    main()
