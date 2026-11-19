#!/usr/bin/env python3
"""Run the frozen E2 paper-normal adaptation across three prepared profiles."""
from __future__ import annotations

import csv
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
import urllib.request
from pathlib import Path

ROOT = Path("/home/chx/projects/tokenflow-repro-paper-candidate")
STAGE = ROOT / "repro/stage3-paper-alignment-v1"
WORKLOADS = STAGE / "workloads"
OUT = STAGE / "e2-paper-normal-v1"
PYTHON = str(ROOT / ".venv/bin/python")
PATCH_SHA256 = "1099c3fc37f782c057ad691af9288f22c537384cabf796ef6c3a1e4feea710fd"
SOURCE_COMMIT = "e754998d6980656707b9628fb5121a4ac4f985df"
PORT = 30000
PROFILES = {
    "plain": "sglang-fork-plain",
    "chunked": "sglang-fork-chunked-2048",
    "tokenflow": "tokenflow-paper-v1-chunked-2048",
}
ORDER = {
    101: ["plain", "chunked", "tokenflow"],
    102: ["chunked", "tokenflow", "plain"],
    103: ["tokenflow", "plain", "chunked"],
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_gpu1_idle() -> None:
    target = subprocess.check_output(
        ["nvidia-smi", "-i", "1", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True
    )
    active = [x.strip() for x in rows.splitlines() if target in x]
    used = int(
        subprocess.check_output(
            ["nvidia-smi", "-i", "1", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
        ).strip()
    )
    if active or used > 200:
        raise RuntimeError(f"GPU1 is busy; refusing to stop or interfere with another task: {active}, {used} MiB")


def check_port_free() -> None:
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", PORT)) == 0:
            raise RuntimeError(f"port {PORT} is occupied; refusing to touch the process")


def wait_for_our_server_to_exit() -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            check_port_free()
            check_gpu1_idle()
            return
        except RuntimeError:
            time.sleep(1)
    check_port_free()
    check_gpu1_idle()


def request(path: str, method: str = "GET") -> bytes:
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}", method=method, data=b"" if method == "POST" else None
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        return response.read()


class ResourceMonitor:
    def __init__(self, path: Path):
        self.path = path
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def run(self) -> None:
        import psutil

        with self.path.open("x") as f:
            writer = csv.writer(f)
            writer.writerow(["unix_s", "gpu0_mib", "gpu0_util", "gpu1_mib", "gpu1_util", "gpu1_watts", "gpu1_temp_c", "host_used_gib", "host_available_gib", "cpu_percent"])
            while not self.stop_event.is_set():
                try:
                    raw = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu", "--format=csv,noheader,nounits"],
                        text=True,
                        timeout=5,
                    )
                    g = [line.split(",") for line in raw.strip().splitlines()]
                    mem = psutil.virtual_memory()
                    writer.writerow([time.time(), g[0][0], g[0][1], g[1][0], g[1][1], g[1][2], g[1][3], (mem.total-mem.available)/2**30, mem.available/2**30, psutil.cpu_percent()])
                    f.flush()
                except Exception as exc:
                    print(json.dumps({"event": "resource_monitor_warning", "error": repr(exc)}), flush=True)
                self.stop_event.wait(1)

    def finish(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)


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


def run_logged(argv: list[str], log_path: Path, timeout: int) -> None:
    with log_path.open("x") as log:
        subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="resume a frozen matrix without overwriting prior artifacts")
    args = parser.parse_args()
    if OUT.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite an existing experiment directory: {OUT}")
    for sub in ["raw", "logs", "server", "env", "smoke", "warmup", "resources"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    patch = subprocess.check_output(["git", "diff"], cwd=ROOT)
    patch_sha = sha256_bytes(patch)
    if commit != SOURCE_COMMIT or patch_sha != PATCH_SHA256:
        raise RuntimeError(f"source identity changed: commit={commit}, patch={patch_sha}")

    index = json.loads((WORKLOADS / "workload-index.json").read_text())
    if index["prompt_corpus_sha256"] != "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4":
        raise RuntimeError("unexpected ShareGPT dataset SHA")
    workload_paths = {}
    for seed in ORDER:
        name = f"e2-paper-normal-adaptation-s{seed}.json"
        path = WORKLOADS / name
        item = next(r for r in index["workloads"] if r["file"] == name)
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"workload changed after freezing: {name}")
        workload_paths[seed] = path
    warm_path = WORKLOADS / "e1-paper-normal-adaptation-s101.json"
    if not warm_path.exists():
        raise FileNotFoundError(warm_path)
    smoke_gate_path = STAGE / "short-smoke-gate/summary.json"
    smoke_gate = json.loads(smoke_gate_path.read_text())
    if not smoke_gate.get("all_profiles_passed"):
        raise RuntimeError("short-output smoke gate is missing or failed; refusing E2 measurement")

    protocol = {
        "protocol": "E2 three-profile paper-normal adaptation",
        "suite": "paper-normal-adaptation",
        "case": "E2",
        "seeds": list(ORDER),
        "profile_order": ORDER,
        "warmup": "E1 paper-normal adaptation seed101, excluded",
        "source_commit": commit,
        "source_patch_sha256": patch_sha,
        "config_matrix_sha256": sha256_file(STAGE / "config-matrix.json"),
        "workload_index_sha256": sha256_file(WORKLOADS / "workload-index.json"),
        "short_smoke_gate_sha256": sha256_file(smoke_gate_path),
        "measurement_workloads": {str(seed): {"file": path.name, "sha256": sha256_file(path)} for seed,path in workload_paths.items()},
        "smoke_requests_per_profile": 9,
        "smoke_passed_all_profiles_before_measurement": True,
        "smoke_and_warmup_excluded": True,
    }
    protocol_path = OUT / "protocol.json"
    if args.resume:
        if not protocol_path.exists():
            raise FileNotFoundError(f"cannot resume without frozen protocol: {protocol_path}")
        old_protocol = json.loads(protocol_path.read_text())
        for key in ["source_commit", "source_patch_sha256", "config_matrix_sha256", "workload_index_sha256", "short_smoke_gate_sha256", "measurement_workloads"]:
            if old_protocol.get(key) != protocol.get(key):
                raise RuntimeError(f"resume protocol mismatch for {key}")
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2))

    progress_path = OUT / "progress.json"
    failures_path = OUT / "failures.json"
    results = json.loads(progress_path.read_text()) if progress_path.exists() else []
    failures = json.loads(failures_path.read_text()) if failures_path.exists() else []
    for seed, order in ORDER.items():
        for short_profile in order:
            profile = PROFILES[short_profile]
            tag = f"E2-normal-{short_profile}-s{seed}"
            server_log = OUT / "server" / f"{tag}.log"
            warm_dir = OUT / "warmup" / tag
            raw_dir = OUT / "raw" / tag
            info_path = OUT / "env" / f"{tag}-server-info.json"
            raw_run_path = raw_dir / "run.json"
            if raw_run_path.exists():
                run_result = json.loads(raw_run_path.read_text())
                if run_result.get("passed") == run_result.get("requests") == 80 and (raw_dir / "analysis.json").exists():
                    if not any(r.get("seed") == seed and r.get("profile") == short_profile for r in results):
                        rows = json.loads((raw_dir / "analysis.json").read_text())
                        if rows:
                            results.append(rows[0])
                            progress_path.write_text(json.dumps(results, indent=2))
                    print(json.dumps({"event": "resume_skip_completed", "tag": tag}), flush=True)
                    continue
                if not any(f.get("tag") == tag for f in failures):
                    failures.append({"tag": tag, "seed": seed, "profile": short_profile, "stage": "existing_partial_run", "run": run_result})
                    failures_path.write_text(json.dumps(failures, indent=2))
                print(json.dumps({"event": "resume_skip_existing_partial", "tag": tag, "passed": run_result.get("passed"), "requested": run_result.get("requests")}), flush=True)
                continue
            monitor = None
            proc = None
            check_gpu1_idle()
            check_port_free()
            monitor = ResourceMonitor(OUT / "resources" / f"{tag}.csv")
            monitor.start()
            with server_log.open("x") as log:
                proc = subprocess.Popen(
                    ["bash", str(STAGE / "launch_profile.sh"), profile],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            try:
                deadline = time.monotonic() + 900
                info = None
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        raise RuntimeError(f"{tag} server exited early: {proc.returncode}; see {server_log}")
                    try:
                        request("/health")
                        info = json.loads(request("/server_info"))
                        break
                    except Exception:
                        time.sleep(2)
                if info is None:
                    raise TimeoutError(f"{tag} server start timed out")
                common = {
                    "dtype": "bfloat16", "tp_size": 1, "mem_fraction_static": 0.8,
                    "max_running_requests": 128, "max_total_num_tokens": 77001,
                    "attention_backend": "triton", "sampling_backend": "pytorch",
                    "enable_deterministic_inference": True, "disable_piecewise_cuda_graph": True,
                }
                expected = {
                    "plain": {"chunked_prefill_size": -1, "enable_hierarchical_cache": False, "kv_offload_policy": "default"},
                    "chunked": {"chunked_prefill_size": 2048, "enable_hierarchical_cache": False, "kv_offload_policy": "default"},
                    "tokenflow": {"chunked_prefill_size": 2048, "enable_hierarchical_cache": True, "kv_offload_policy": "paper_v1"},
                }[short_profile]
                checks = {**common, **expected}
                failed = {k: (info.get(k), v) for k,v in checks.items() if info.get(k) != v}
                if failed:
                    raise AssertionError(f"{tag} server-info mismatch: {failed}")
                info_path.write_text(json.dumps(info, indent=2))
                (OUT / "env" / f"{tag}-environment.json").write_text(json.dumps({
                    "profile": profile,
                    "SGLANG_ENABLE_SYNC_CACHE": "1" if short_profile == "tokenflow" else "0",
                    "TOKENFLOW_REPRO_FENCE_WRITES": "1" if short_profile == "tokenflow" else "unset",
                    "source_commit": commit,
                    "source_patch_sha256": patch_sha,
                    "checks": checks,
                }, indent=2))
                command = STAGE / "preflight" / f"{profile}-command.txt"
                (OUT / "env" / f"{tag}-command.txt").write_text(command.read_text())

                run_logged([PYTHON, "repro/scripts/replay.py", "run", "--file", str(warm_path), "--out", str(warm_dir)], OUT / "logs" / f"{tag}-warmup.log", 1200)
                warm_result = json.loads((warm_dir / "run.json").read_text())
                if warm_result.get("passed") != warm_result.get("requests"):
                    raise RuntimeError(f"{tag} warmup incomplete: {warm_result}")
                request("/flush_cache", "POST")

                measurement_start = time.time()
                if raw_dir.exists():
                    raise FileExistsError(raw_dir)
                run_logged([PYTHON, "repro/scripts/replay.py", "run", "--file", str(workload_paths[seed]), "--out", str(raw_dir)], OUT / "logs" / f"{tag}-client.log", 2400)
                run_result = json.loads((raw_dir / "run.json").read_text())
                if run_result.get("passed") != 80 or run_result.get("workload_sha256") != sha256_file(workload_paths[seed]):
                    raise RuntimeError(f"{tag} measurement failed or workload hash changed: {run_result}")
                analysis_path = raw_dir / "analysis.json"
                run_logged([PYTHON, "repro/scripts/analyze.py", str(raw_dir), "--speed", "20", "--out", str(analysis_path)], OUT / "logs" / f"{tag}-analysis.log", 300)
                rows = json.loads(analysis_path.read_text())
                row = rows[0]
                row.update({"case": "E2", "suite": "paper-normal-adaptation", "profile": short_profile, "seed": seed, "measurement_start_unix_s": measurement_start, "short_smoke_gate_passed": True, "warmup_passed": warm_result["passed"]})
                analysis_path.write_text(json.dumps([row], indent=2))
                results.append(row)
                progress_path.write_text(json.dumps(results, indent=2))
                print(json.dumps({"event": "measurement_complete", **row}), flush=True)
            except Exception as exc:
                failure = {"tag": tag, "seed": seed, "profile": short_profile, "stage": "measurement", "error": repr(exc)}
                if raw_run_path.exists():
                    failure["run"] = json.loads(raw_run_path.read_text())
                failures.append(failure)
                failures_path.write_text(json.dumps(failures, indent=2))
                print(json.dumps({"event": "measurement_failed_continue", **failure}), flush=True)
            finally:
                stop_ours(proc)
                if monitor:
                    monitor.finish()
            wait_for_our_server_to_exit()

    profiles_summary = {}
    for key in PROFILES:
        rows = [r for r in results if r["profile"] == key]
        profile_summary = {"runs": len(rows), "failures": sum(f.get("profile") == key for f in failures)}
        if rows:
            profile_summary.update({
                "mean_raw_output_tok_s": statistics.mean(r["raw_output_tok_s"] for r in rows),
                "mean_effective_tok_s": statistics.mean(r["effective_tok_s"] for r in rows),
                "mean_ttft_s": statistics.mean(r["ttft_mean_s"] for r in rows),
                "mean_p99_ttft_s": statistics.mean(r["ttft_p99_s"] for r in rows),
                "per_seed": [{k:r[k] for k in ["seed", "raw_output_tok_s", "effective_tok_s", "ttft_mean_s", "ttft_p99_s", "completed"] if k in r} for r in rows],
            })
        profiles_summary[key] = profile_summary
    summary = {
        "case": "E2",
        "suite": "paper-normal-adaptation",
        "attempted_runs": len(results) + len(failures),
        "completed_runs": len(results),
        "failed_runs": len(failures),
        "requested_requests": sum(r["requested"] for r in results) + sum(f.get("run", {}).get("requests", 0) for f in failures),
        "completed_requests": sum(r["completed"] for r in results) + sum(f.get("run", {}).get("passed", 0) for f in failures),
        "fully_analyzed_requests": sum(r["requested"] for r in results),
        "partial_run_completed_requests": sum(f.get("run", {}).get("passed", 0) for f in failures),
        "profile_summary_arithmetic_mean_of_per_run_metrics": profiles_summary,
        "performance_interpretation": "adaptation comparison only; not a confirmed reproduction of the paper's undisclosed original commit or normal-distribution parameters",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"event": "matrix_complete", **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
