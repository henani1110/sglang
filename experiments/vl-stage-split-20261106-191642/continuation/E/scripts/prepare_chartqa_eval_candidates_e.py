#!/usr/bin/env python3
"""Build a fixed, human-authored ChartQA calibration/heldout manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    raw = args.input_manifest.read_bytes()
    source_rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    rows = []
    for item in source_rows:
        if item.get("benchmark_subset") != "ChartQA-H":
            raise SystemExit("candidate manifest contains a non-human ChartQA subset")
        split = item.get("split")
        if split not in {"train", "val"}:
            raise SystemExit(f"unexpected split {split!r}")
        rows.append({
            "sample_id": item["sample_id"],
            "quality_split": "calibration" if split == "train" else "heldout",
            "source_split": split,
            "image_id": item["image_id"],
            "image_relative_path": f"images/{split}/{item['image_id']}",
            "image_sha256": item["image_sha256"],
            "question": item["question"],
            "answer": item["answer"],
            "source_name_guess": item["source_name_guess"],
            "source_name_status": item["source_name_status"],
            "rights_status": item["rights_status"],
            "redistributable": False,
            "formal_quality_candidate": False,
        })

    rows.sort(key=lambda row: row["sample_id"])
    counts = {name: sum(row["quality_split"] == name for row in rows)
              for name in ("calibration", "heldout")}
    if counts != {"calibration": 80, "heldout": 40}:
        raise SystemExit(f"expected train/val human 80/40; got {counts}")
    image_sets = {
        name: {row["image_id"] for row in rows if row["quality_split"] == name}
        for name in ("calibration", "heldout")
    }
    if image_sets["calibration"] & image_sets["heldout"]:
        raise SystemExit("calibration and heldout image sets overlap")
    if len(image_sets["calibration"]) != 40 or len(image_sets["heldout"]) != 20:
        raise SystemExit("expected 40 train and 20 val unique images")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "quality-candidates.jsonl"
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                    for row in rows).encode("utf-8")
    manifest.write_bytes(data)
    summary = {
        "dataset": "ChartQA-H OECD-prefix local exploratory subset",
        "candidate_manifest": manifest.name,
        "candidate_manifest_sha256": hashlib.sha256(data).hexdigest(),
        "source_candidate_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "sample_counts": counts,
        "unique_images": {name: len(values) for name, values in image_sets.items()},
        "image_overlap_count": len(image_sets["calibration"] & image_sets["heldout"]),
        "test_annotations_read": False,
        "flickr_api_or_url_used": False,
        "eligible_quality_items_added": 0,
        "formal_quality_candidate": False,
        "rights_status": "all_source_license_pending; local exploratory evaluation only",
        "heldout_policy": "official val subset frozen; do not use for prompt or policy tuning",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
