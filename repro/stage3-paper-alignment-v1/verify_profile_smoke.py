#!/usr/bin/env python3
"""Check short deterministic token outputs across all three profiles before E2."""
from __future__ import annotations

import json
import hashlib
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path("/home/chx/projects/tokenflow-repro-paper-candidate")
STAGE = ROOT / "repro/stage3-paper-alignment-v1"
OUT = STAGE / "short-smoke-gate"
PYTHON = str(ROOT / ".venv/bin/python")
PORT = 30000
SOURCE_COMMIT = "e754998d6980656707b9628fb5121a4ac4f985df"
PATCH_SHA256 = "1099c3fc37f782c057ad691af9288f22c537384cabf796ef6c3a1e4feea710fd"
PROFILES = {
    "plain": "sglang-fork-plain",
    "chunked": "sglang-fork-chunked-2048",
    "tokenflow": "tokenflow-paper-v1-chunked-2048",
}


def assert_idle() -> None:
    target = subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True)
    active = [row for row in apps.splitlines() if target in row]
    mem = int(subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
    if active or mem > 200:
        raise RuntimeError(f"GPU1 is busy: {active}, {mem} MiB")
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            raise RuntimeError(f"port {PORT} is busy; refusing to touch its process")


def wait_idle() -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            assert_idle()
            return
        except RuntimeError:
            time.sleep(1)
    assert_idle()


def get(path: str) -> bytes:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=10) as response:
        return response.read()


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
    if OUT.exists():
        raise FileExistsError(OUT)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    patch_sha = hashlib.sha256(subprocess.check_output(["git", "diff"], cwd=ROOT)).hexdigest()
    if commit != SOURCE_COMMIT or patch_sha != PATCH_SHA256:
        raise RuntimeError(f"source identity changed: commit={commit}, patch={patch_sha}")
    for sub in ["logs", "smoke", "server-info"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    results = []
    reference = None
    for key, profile in PROFILES.items():
        wait_idle()
        log_path = OUT / "logs" / f"{key}.log"
        smoke_dir = OUT / "smoke" / key
        with log_path.open("x") as log:
            proc = subprocess.Popen(["bash", str(STAGE / "launch_profile.sh"), profile], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 900
            info = None
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"{key} server exited: {proc.returncode}; inspect {log_path}")
                try:
                    get("/health")
                    info = json.loads(get("/server_info"))
                    break
                except Exception:
                    time.sleep(2)
            if info is None:
                raise TimeoutError(f"{key} server startup timed out")
            checks = {
                "dtype": "bfloat16", "tp_size": 1, "mem_fraction_static": 0.8,
                "max_running_requests": 128, "max_total_num_tokens": 77001,
                "attention_backend": "triton", "sampling_backend": "pytorch",
                "enable_deterministic_inference": True, "disable_piecewise_cuda_graph": True,
                "chunked_prefill_size": -1 if key == "plain" else 2048,
                "enable_hierarchical_cache": key == "tokenflow",
                "kv_offload_policy": "paper_v1" if key == "tokenflow" else "default",
            }
            failed = {field: (info.get(field), expected) for field, expected in checks.items() if info.get(field) != expected}
            if failed:
                raise AssertionError(f"{key} server-info mismatch: {failed}")
            (OUT / "server-info" / f"{key}.json").write_text(json.dumps(info, indent=2))
            with (OUT / "logs" / f"{key}-smoke.log").open("x") as smoke_log:
                subprocess.run([PYTHON, "repro/stage3-paper-alignment-v1/smoke_stream.py", str(smoke_dir)], cwd=ROOT, stdout=smoke_log, stderr=subprocess.STDOUT, check=True, timeout=600)
            summary = json.loads((smoke_dir / "summary.json").read_text())
            if summary.get("passed") != 9:
                raise RuntimeError(f"{key} smoke failed: {summary}")
            requests = [json.loads(row) for row in (smoke_dir / "requests.jsonl").read_text().splitlines()]
            token_ids = [row["events"][-1]["data"]["output_ids"] for row in requests]
            matches = reference is None or token_ids == reference
            if reference is None:
                reference = token_ids
                (OUT / "reference-token-ids.json").write_text(json.dumps(reference))
            results.append({"profile": key, "passed": summary["passed"], "output_tokens": summary["output_tokens"], "matches_first_profile": matches})
            print(json.dumps(results[-1]), flush=True)
            if not matches:
                raise RuntimeError(f"short deterministic output mismatch in {key}; abort before E2 matrix")
        finally:
            stop_ours(proc)
        wait_idle()
    summary = {"all_profiles_passed": all(r["passed"] == 9 and r["matches_first_profile"] for r in results), "profiles": results}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
