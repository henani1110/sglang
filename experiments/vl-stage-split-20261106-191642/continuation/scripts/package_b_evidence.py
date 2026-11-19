#!/usr/bin/env python3
"""Collect B-stage raw evidence on the SSD and produce a non-overwriting archive."""
from __future__ import annotations

import shutil
import subprocess
import tarfile
from pathlib import Path


HOME = Path("/home/chx/project/vl-stage-split-20261106-191642")
SSD = Path("/media/chx/Win10/vl-stage-split-20261106-191642")
CONT = HOME / "continuation"
DEST = SSD / "continuation/raw/b_evidence_stage"
ARCHIVE = SSD / "continuation/raw/B-evidence.tar.gz"
LABELS = ("B-N-r0", "B-N-r0-retry1", "B-S-r0", "B-S-r1-2", "B-S-cancel-r0")
VALID_LABELS = LABELS[1:]


def copy_if_present(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    if ARCHIVE.exists():
        raise SystemExit(f"refusing to overwrite existing archive: {ARCHIVE}")
    for sub in ("results", "logs", "runlogs", "scripts", "patches", "source-snapshots", "manifests"):
        (DEST / sub).mkdir(parents=True, exist_ok=True)

    for label in LABELS:
        for name in (f"{label}.json",):
            copy_if_present(SSD / "results" / name, DEST / "results" / name)
        for suffix in (
            "client-events.jsonl", "diagnostics.jsonl", "equivalence.jsonl",
            "scheduler-events.jsonl", "server-events.jsonl",
            "server-events.jsonl.async-meta.json", "resources.csv",
            "gpu-pcie-start.txt", "gpu-pcie-ready.txt",
        ):
            name = f"{label}-{suffix}"
            copy_if_present(SSD / "logs" / name, DEST / "logs" / name)
        for suffix in (
            "server.log", "client.stdout.log", "monitor.log", "runner-status.txt",
            "server.pid", "server.pgid", "monitor.pid",
        ):
            name = f"{label}-{suffix}"
            copy_if_present(HOME / "logs" / name, DEST / "runlogs" / name)

    for src, relative in (
        (CONT / "logs/continuation-run-ledger.csv", "manifests/continuation-run-ledger.csv"),
        (HOME / "logs/service-ledger.csv", "manifests/service-ledger.csv"),
        (CONT / "patches/scheduler_encoder_only_split.diff", "patches/scheduler_encoder_only_split.diff"),
        (CONT / "patches/async_probe.diff", "patches/async_probe.diff"),
    ):
        copy_if_present(src, DEST / relative)
    for src in (CONT / "scripts").glob("*"):
        if src.is_file():
            copy_if_present(src, DEST / "scripts" / src.name)
    for src in (CONT / "source-snapshots").rglob("*"):
        if src.is_file():
            copy_if_present(src, DEST / "source-snapshots" / src.relative_to(CONT / "source-snapshots"))

    old = Path("/home/chx/project/vl-encode-pilot-20261106-085428")
    py = old / "env/serve311/bin/python"
    runtime = subprocess.check_output(
        [str(py), "-c", 'import vllm, torch; print("vllm="+vllm.__version__); print("torch="+torch.__version__); print("cuda="+str(torch.version.cuda))'],
        text=True,
    )
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id", "--format=csv,noheader"],
        text=True,
    )
    (DEST / "manifests/runtime.txt").write_text(
        "host=chx-gpu-4090-2\n"
        "gpu0_uuid=GPU-7d51d535-c0d2-577a-2a08-d6fbdd9b93a8\n"
        "gpu0_link=PCIe Gen4 x16 (from ready/start nvidia-smi -q captures)\n"
        + runtime + gpu
    )

    missing: list[str] = []
    for label in VALID_LABELS:
        for required in (DEST / "results" / f"{label}.json",):
            if not required.is_file():
                missing.append(str(required.relative_to(DEST)))
        for suffix in ("client-events.jsonl", "scheduler-events.jsonl", "server-events.jsonl"):
            required = DEST / "logs" / f"{label}-{suffix}"
            if not required.is_file():
                missing.append(str(required.relative_to(DEST)))
    if missing:
        raise SystemExit("missing required evidence: " + ", ".join(missing))
    with tarfile.open(ARCHIVE, "w:gz") as tf:
        tf.add(DEST, arcname="b_evidence")
    print(f"archive={ARCHIVE} bytes={ARCHIVE.stat().st_size}")
    print(f"files={sum(1 for p in DEST.rglob('*') if p.is_file())}")
    print("missing_optional_logs=" + repr([
        (label, suffix) for label in LABELS
        for suffix in ("diagnostics.jsonl", "equivalence.jsonl", "resources.csv", "gpu-pcie-start.txt", "gpu-pcie-ready.txt")
        if not (DEST / "logs" / f"{label}-{suffix}").exists()
    ]))
    print(subprocess.check_output(["sha256sum", str(ARCHIVE)], text=True).strip())


if __name__ == "__main__":
    main()
