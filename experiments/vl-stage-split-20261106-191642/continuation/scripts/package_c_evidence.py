#!/usr/bin/env python3
"""Package C-stage raw artifacts from the remote project and SSD without caches/weights."""
from __future__ import annotations

import hashlib
import json
import os
import tarfile
import time
from pathlib import Path

PROJECT = Path("/home/chx/project/vl-stage-split-20261106-191642")
SSD = Path("/media/chx/Win10/vl-stage-split-20261106-191642")
ARCHIVE = SSD / "continuation/raw/C-evidence.tar.gz"


def candidates() -> list[tuple[Path, str]]:
    out: dict[str, Path] = {}

    def add(path: Path, arc: str | None = None) -> None:
        if not path.exists():
            return
        key = arc or str(path.relative_to(SSD))
        out[key] = path

    for pattern in ("C*.json", "C*.md"):
        for path in (SSD / "results").glob(pattern):
            add(path)
    for path in (SSD / "logs").glob("C*"):
        add(path)
    for path in (PROJECT / "logs").glob("C*"):
        add(path, "project-logs/" + path.name)
    add(PROJECT / "logs/service-ledger.csv", "project-logs/service-ledger.csv")
    for path in (PROJECT / "continuation/logs").glob("C*"):
        add(path, "project-continuation-logs/" + path.name)
    for path in (PROJECT / "continuation/patches").glob("*"):
        add(path, "patches/" + path.name)
    for path in (PROJECT / "continuation/scripts").glob("*.py"):
        add(path, "scripts/" + path.name)
    for path in (PROJECT / "continuation/scripts").glob("*.sh"):
        add(path, "scripts/" + path.name)
    add(PROJECT / "scripts/trace_client.py", "scripts/trace_client.py")
    for path in (SSD / "continuation/manifests").glob("C*"):
        if path.name != "C-evidence-file-manifest.json":
            add(path)
    for name in ("datasets-512", "datasets-768"):
        add(SSD / name)
    for path in (PROJECT / "pilot_control.json", PROJECT / "host-info.txt", PROJECT / "software-info.txt"):
        add(path, "project-metadata/" + path.name)

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project": str(PROJECT),
        "ssd": str(SSD),
        "policy": "Include C results/logs, F calibration, datasets/manifests, run scripts and applied patch sources; exclude model weights and runtime caches.",
        "files": [{"archive_path": k, "source_path": str(v), "size_bytes": v.stat().st_size if v.is_file() else None} for k, v in sorted(out.items())],
    }
    manifest_path = SSD / "continuation/manifests/C-evidence-file-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    out[str(manifest_path.relative_to(SSD))] = manifest_path
    return [(path, arc) for arc, path in sorted(out.items())]


def main() -> None:
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ARCHIVE.with_suffix(ARCHIVE.suffix + ".tmp")
    file_rows = []
    with tarfile.open(tmp, "w:gz", compresslevel=6) as tar:
        for path, arcname in candidates():
            tar.add(path, arcname=arcname, recursive=True)
            if path.is_file():
                file_rows.append((arcname, path.stat().st_size))
    os.replace(tmp, ARCHIVE)
    h = hashlib.sha256()
    with ARCHIVE.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    result = {
        "archive": str(ARCHIVE),
        "bytes": ARCHIVE.stat().st_size,
        "sha256": h.hexdigest(),
        "regular_file_count": len(file_rows),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    digest_path = ARCHIVE.with_suffix(ARCHIVE.suffix + ".sha256")
    digest_path.write_text(f"{result['sha256']}  {ARCHIVE.name}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
