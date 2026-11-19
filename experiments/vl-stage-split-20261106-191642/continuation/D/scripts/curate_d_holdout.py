#!/usr/bin/env python3
"""Create a visually de-duplicated, hash-verified 32-image D holdout."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


SELECTION = {
    "commons-cc0": [
        "cc0-00-d0e12317a547", "cc0-01-f7be9b25783d", "cc0-02-74dec2e614c6",
        "cc0-03-5ef75aaf062c", "cc0-06-79495392e2c8", "cc0-07-1462caaaf9e1",
        "cc0-08-a899491636a3", "cc0-09-aa4f074784e8", "cc0-11-1d4867509d57",
        "cc0-12-391fcf7adff7", "cc0-13-abb3cc2b706f",
        "cc0-16-f36c6c24da3f", "cc0-17-e798fbbd08cd", "cc0-19-2b2cb79d1452",
        "cc0-21-65657cc12c60", "cc0-22-9075b5c9f3ff", "cc0-26-d8c95254b483",
        "cc0-27-321b1421e6ef", "cc0-28-eaf3affddfa5", "cc0-30-1f298b4e7825",
        "cc0-31-ff8889385b2a",
    ],
    "commons-cc0-expansion": [
        "cc0-00-17f3d1beff40", "cc0-01-47c3980d0d1b", "cc0-03-0aee8fef71a4",
        "cc0-04-75ad201ec925", "cc0-05-a9b3268618cb",
    ],
    "commons-cc0-expansion-2": [
        "cc0-00-317f4daebbee", "cc0-01-3e46fb5858d3", "cc0-02-e7f9295a6e19",
        "cc0-03-76b40df2d44a", "cc0-04-1d2cf14d6ab5",
        "cc0-05-225b40a6fb76",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> bytes:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    source_manifest_hashes: dict[str, str] = {}
    source_rows: dict[str, tuple[str, Path, dict[str, Any]]] = {}

    for collection, ids in SELECTION.items():
        collection_root = args.datasets_root / collection
        manifest_path = collection_root / "commons-source-manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        source_manifest_hashes[collection] = hashlib.sha256(manifest_bytes).hexdigest()
        manifest = json.loads(manifest_bytes)
        by_id = {row["image_id"]: row for row in manifest.get("images", [])}
        for image_id in ids:
            row = by_id.get(image_id)
            if row is None:
                raise RuntimeError(f"missing selected image {collection}/{image_id}")
            source_rows[image_id] = (collection, collection_root, row)

    if len(source_rows) != 32:
        raise RuntimeError(f"selection has {len(source_rows)} unique source IDs, expected 32")

    local_rows: list[dict[str, Any]] = []
    remote_rows: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    source_hashes: set[str] = set()
    for index, (image_id, (collection, collection_root, source)) in enumerate(source_rows.items(), 1):
        if source.get("license") not in ("CC0", "CC-Zero"):
            raise RuntimeError(f"non-CC0 source in selection: {source['title']}: {source.get('license')}")
        if "/publicdomain/zero/1.0" not in str(source.get("license_url", "")):
            raise RuntimeError(f"missing CC0 legal URL: {source['title']}")
        source_sha = str(source["source_download_sha256"])
        if source_sha in source_hashes:
            raise RuntimeError(f"duplicate source bytes: {source['title']}")
        source_hashes.add(source_sha)
        source_name = Path(source["source_download_path"]).name
        local_source = collection_root / "source_downloads" / source_name
        if not local_source.is_file() or sha256(local_source) != source_sha:
            raise RuntimeError(f"source download hash mismatch: {local_source}")

        holdout_id = f"holdout-{index:02d}"
        local_processed: dict[str, dict[str, Any]] = {}
        remote_processed: dict[str, dict[str, Any]] = {}
        for edge in (512, 768):
            source_info = source["processed"][str(edge)]
            source_path = collection_root / "images" / str(edge) / Path(source_info["path"]).name
            if not source_path.is_file() or sha256(source_path) != source_info["sha256"]:
                raise RuntimeError(f"processed source hash mismatch: {source_path}")
            filename = f"{holdout_id}.png"
            local_path = args.output_root / "images" / str(edge) / filename
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, local_path)
            processed_sha = sha256(local_path)
            if processed_sha != source_info["sha256"]:
                raise RuntimeError(f"copy hash mismatch: {local_path}")
            shared = {"sha256": processed_sha, "width": edge, "height": edge, "bytes": local_path.stat().st_size}
            local_processed[str(edge)] = {**shared, "path": str(local_path)}
            remote_path = f"{args.remote_root.rstrip('/')}/images/{edge}/{filename}"
            remote_processed[str(edge)] = {**shared, "path": remote_path, "local_path": str(local_path)}

        category = str(source.get("search_phrase") or "uncategorized")
        category_counts[category] += 1
        common = {
            "image_id": holdout_id,
            "source_image_id": image_id,
            "source_collection": collection,
            "source_manifest_sha256": source_manifest_hashes[collection],
            "title": source["title"],
            "description_url": source.get("description_url"),
            "original_url": source.get("original_url"),
            "license": source["license"],
            "license_url": source["license_url"],
            "artist_credit": source.get("artist_credit", ""),
            "description": source.get("description", ""),
            "search_phrase": source.get("search_phrase"),
            "source_download_sha256": source_sha,
            "source_download_bytes": source.get("source_download_bytes"),
            "source_download_remote_path": source["source_download_path"],
            "source_download_local_path": str(local_source),
            "processed": None,
        }
        local_rows.append({**common, "processed": local_processed})
        remote_rows.append({**common, "processed": remote_processed})

    common_manifest = {
        "schema_version": 1,
        "dataset_id": "D-holdout-32-curated-v1",
        "status": "complete",
        "required_count": 32,
        "selected_count": 32,
        "license_policy": "each source row has CC0/CC-Zero LicenseShortName and CC0 1.0 LicenseUrl",
        "selection_note": "visually reviewed contact sheets; removed repeated Old Judge, Lincoln, church, Irinka, lion-art, and near-identical Hamzanama images; retained all acquisition attempts separately",
        "processor_grids": [512, 768],
        "unique_source_download_sha256": len(source_hashes),
        "source_manifests": source_manifest_hashes,
        "search_topic_counts": dict(category_counts),
    }
    local_manifest = {**common_manifest, "images": local_rows}
    remote_manifest = {**common_manifest, "images": remote_rows}
    atomic_json(args.output_root / "holdout-selected-manifest.local.json", local_manifest)
    remote_bytes = atomic_json(args.output_root / "holdout-selected-manifest.json", remote_manifest)
    remote_sha = hashlib.sha256(remote_bytes).hexdigest()
    (args.output_root / "holdout-selected-manifest.json.sha256").write_text(
        f"{remote_sha}  holdout-selected-manifest.json\n"
    )
    print(json.dumps({"status": "complete", "selected": len(remote_rows), "unique_source_hashes": len(source_hashes), "category_counts": dict(category_counts), "remote_manifest_sha256": remote_sha}, ensure_ascii=False))


if __name__ == "__main__":
    main()
