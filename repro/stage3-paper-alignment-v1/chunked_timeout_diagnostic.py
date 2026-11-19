#!/usr/bin/env python3
"""Replay two exact E2 chunked-timeout requests alone after standard warmup."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/chx/projects/tokenflow-repro-paper-candidate")
STAGE = ROOT / "repro/stage3-paper-alignment-v1"
OUT = STAGE / "chunked-timeout-diagnostic-v1"
PYTHON = str(ROOT / ".venv/bin/python")
SOURCE_COMMIT = "e754998d6980656707b9628fb5121a4ac4f985df"
PATCH_SHA256 = "1099c3fc37f782c057ad691af9288f22c537384cabf796ef6c3a1e4feea710fd"

sys.path.insert(0, str(STAGE))
import stage3_e2_driver as driver  # noqa: E402


def source_identity() -> tuple[str, str]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(["git", "diff"], cwd=ROOT)
    return commit, hashlib.sha256(diff).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    for name in ["server", "env", "warmup", "raw", "logs"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    commit, patch = source_identity()
    if (commit, patch) != (SOURCE_COMMIT, PATCH_SHA256):
        raise RuntimeError(f"candidate source changed: {(commit, patch)}")

    profile = "sglang-fork-chunked-2048"
    log_path = OUT / "server" / "server.log"
    driver.check_gpu1_idle()
    driver.check_port_free()
    with log_path.open("x") as log:
        proc = subprocess.Popen(
            ["bash", str(STAGE / "launch_profile.sh"), profile],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    results = []
    try:
        deadline = time.monotonic() + 900
        info = None
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited early: {proc.returncode}")
            try:
                driver.request("/health")
                info = json.loads(driver.request("/server_info"))
                break
            except Exception:
                time.sleep(2)
        if info is None:
            raise TimeoutError("server start timed out")
        expected = {
            "dtype": "bfloat16",
            "tp_size": 1,
            "mem_fraction_static": 0.8,
            "max_running_requests": 128,
            "max_total_num_tokens": 77001,
            "attention_backend": "triton",
            "sampling_backend": "pytorch",
            "enable_deterministic_inference": True,
            "disable_piecewise_cuda_graph": True,
            "chunked_prefill_size": 2048,
            "enable_hierarchical_cache": False,
            "kv_offload_policy": "default",
        }
        mismatches = {k: (info.get(k), v) for k, v in expected.items() if info.get(k) != v}
        if mismatches:
            raise AssertionError(f"server info mismatch: {mismatches}")
        (OUT / "env" / "server-info.json").write_text(json.dumps(info, indent=2))
        (OUT / "env" / "protocol.json").write_text(json.dumps({
            "profile": profile,
            "source_commit": commit,
            "source_patch_sha256": patch,
            "warmup": "E1 paper-normal adaptation seed101, 60 requests, excluded",
            "diagnostic_requests": [{"seed": 101, "id": 9}, {"seed": 102, "id": 66}],
            "purpose": "separate request-content/length effects from high-concurrency burst/retraction effects; not performance data",
        }, indent=2))

        warm_path = STAGE / "workloads/e1-paper-normal-adaptation-s101.json"
        driver.run_logged(
            [PYTHON, "repro/scripts/replay.py", "run", "--file", str(warm_path), "--out", str(OUT / "warmup" / "e1-s101")],
            OUT / "logs" / "warmup.log",
            1200,
        )
        warm = json.loads((OUT / "warmup" / "e1-s101" / "run.json").read_text())
        if warm.get("passed") != 60:
            raise RuntimeError(f"warmup incomplete: {warm}")
        driver.request("/flush_cache", "POST")

        for seed, request_id in [(101, 9), (102, 66)]:
            source = json.loads((STAGE / "workloads" / f"e2-paper-normal-adaptation-s{seed}.json").read_text())
            row = source["requests"][request_id]
            one_request = {**{k: v for k, v in source.items() if k != "requests"}, "requests": [row]}
            workload = OUT / "env" / f"s{seed}-rid{request_id}.json"
            workload.write_text(json.dumps(one_request))
            tag = f"s{seed}-rid{request_id}"
            raw = OUT / "raw" / tag
            with (OUT / "logs" / f"{tag}.log").open("x") as log:
                completed = subprocess.run(
                    [PYTHON, "repro/scripts/replay.py", "run", "--file", str(workload), "--out", str(raw)],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=2400,
                )
            if not (raw / "run.json").exists():
                raise RuntimeError(f"replay produced no run.json for {tag}: returncode={completed.returncode}")
            run = json.loads((raw / "run.json").read_text())
            result = {"seed": seed, "request_id": request_id, "input_tokens": len(row["input_ids"]), "requested_output_tokens": row["max_new_tokens"], "returncode": completed.returncode, **run}
            results.append(result)
            (OUT / "progress.json").write_text(json.dumps(results, indent=2))
            print(json.dumps({"event": "single_request_complete", **result}), flush=True)
    finally:
        driver.stop_ours(proc)
    driver.wait_for_our_server_to_exit()
    after = source_identity()
    if after != (commit, patch):
        raise RuntimeError(f"candidate source changed during diagnostic: {after}")
    (OUT / "summary.json").write_text(json.dumps({
        "requested": len(results),
        "completed": sum(r.get("passed", 0) == 1 for r in results),
        "runs": results,
        "interpretation": "This isolates each failed workload item from burst concurrency; it does not locate the scheduler state that stalls only under load.",
    }, indent=2))
    print(json.dumps({"event": "diagnostic_complete", "runs": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
