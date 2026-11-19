#!/usr/bin/env python3
"""Extract only manifest-selected ChartQA train/val images and verify hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

PINNED_ARCHIVE_SHA256 = "1bf310e5a51101681495c4a24f4f29d22c4f70b52df24e2e4feb0d79cae3c160"
PINNED_ARCHIVE_BYTES = 875370872


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_member(row: dict) -> str:
    split = row.get("split")
    image_id = row.get("image_id")
    member = row.get("image_member")
    if split not in {"train", "val"}:
        raise ValueError(f"unexpected split: {split!r}")
    if not isinstance(image_id, str) or PurePosixPath(image_id).name != image_id:
        raise ValueError(f"unsafe image_id: {image_id!r}")
    expected = f"ChartQA Dataset/{split}/png/{image_id}"
    if member != expected or ".." in PurePosixPath(member).parts:
        raise ValueError(f"unexpected ZIP member: {member!r}")
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-archive-sha", action="store_true",
                        help="Skip full archive SHA verification when already verified by a pinned acquisition step.")
    args = parser.parse_args()

    if args.archive.stat().st_size != PINNED_ARCHIVE_BYTES:
        raise SystemExit("pinned archive byte-size mismatch")
    archive_sha = None if args.skip_archive_sha else sha256_file(args.archive)
    if archive_sha is not None and archive_sha != PINNED_ARCHIVE_SHA256:
        raise SystemExit(f"pinned archive SHA-256 mismatch: {archive_sha}")

    manifest_bytes = args.manifest.read_bytes()
    rows = [json.loads(line) for line in manifest_bytes.splitlines() if line.strip()]
    selected: dict[str, dict] = {}
    for row in rows:
        member = safe_member(row)
        if row.get("archive_sha256_pinned") != PINNED_ARCHIVE_SHA256:
            raise SystemExit("manifest archive identity mismatch")
        if not isinstance(row.get("image_sha256"), str) or len(row["image_sha256"]) != 64:
            raise SystemExit(f"missing image SHA-256: {row.get('sample_id')}")
        previous = selected.get(member)
        if previous is not None and previous["image_sha256"] != row["image_sha256"]:
            raise SystemExit(f"conflicting hashes for {member}")
        selected[member] = row

    args.output_dir.mkdir(parents=True, exist_ok=True)
    materialized: list[dict] = []
    with zipfile.ZipFile(args.archive) as zf:
        for member, row in sorted(selected.items()):
            raw = zf.read(member)
            digest = hashlib.sha256(raw).hexdigest()
            if digest != row["image_sha256"]:
                raise SystemExit(f"image SHA-256 mismatch for {member}: {digest}")
            split = row["split"]
            out = args.output_dir / "images" / split / row["image_id"]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(raw)
            materialized.append({
                "split": split,
                "image_id": row["image_id"],
                "image_member": member,
                "path": out.relative_to(args.output_dir).as_posix(),
                "bytes": len(raw),
                "sha256": digest,
            })

    summary = {
        "dataset": "ChartQA-H",
        "archive_sha256": archive_sha or PINNED_ARCHIVE_SHA256,
        "archive_sha_verified_here": archive_sha is not None,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_questions": len(rows),
        "unique_selected_images": len(materialized),
        "selected_images": materialized,
        "test_annotation_read": False,
        "flickr_api_or_url_used": False,
        "rights_status": "local_candidate_materialization_only_per_image_source_license_pending",
        "redistributable": False,
    }
    (args.output_dir / "extraction-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "selected_images"},
                     ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
