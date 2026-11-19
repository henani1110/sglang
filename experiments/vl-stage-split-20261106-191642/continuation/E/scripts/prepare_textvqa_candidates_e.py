#!/usr/bin/env python3
"""Build a deterministic TextVQA candidate manifest joined to Open Images V3 metadata.

This script does not download images or mark media as cleared. Open Images V3
warns that its per-image CC metadata should be checked against the original page,
so rows remain pending until that source-level check is recorded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import tarfile
from collections import defaultdict
from pathlib import Path

OI_V3_README = "https://github.com/openimages/dataset/blob/main/READMEV3.md"
TEXTVQA_LICENSE = "https://creativecommons.org/licenses/by/4.0/"
TEXTVQA_PAGE = "https://textvqa.org/dataset/"
OPENIMAGES_CC_BY_2 = "https://creativecommons.org/licenses/by/2.0/"


def normalize_image_id(value: object) -> str:
    s = str(value).strip().lower()
    try:
        return f"{int(s, 16):016x}"
    except ValueError:
        return s.zfill(16)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_questions(path: Path, source_split: str) -> dict[str, list[dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in payload.get("data", []):
        row = dict(item)
        row["_source_split"] = source_split
        row["_normalized_image_id"] = normalize_image_id(item.get("image_id", ""))
        grouped[str(row["_normalized_image_id"])].append(row)
    return grouped


def load_openimages_rows(archive: Path, wanted: set[str]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with tarfile.open(archive, mode="r:gz") as tf:
        for split in ("train", "validation", "test"):
            name = f"2017_11/{split}/images.csv"
            member = tf.getmember(name)
            stream = tf.extractfile(member)
            if stream is None:
                raise RuntimeError(f"cannot read {name}")
            text = io.TextIOWrapper(stream, encoding="utf-8", newline="")
            for row in csv.DictReader(text):
                image_id = normalize_image_id(row.get("ImageID", ""))
                if image_id in wanted:
                    row["_metadata_split"] = split
                    result[image_id] = row
            text.close()
    return result


def is_allowed_metadata_license(value: str) -> bool:
    return value.strip().rstrip("/").lower() == OPENIMAGES_CC_BY_2.rstrip("/").lower()


def canonical_question_sha(item: dict[str, object], image_id: str) -> str:
    payload = {
        "dataset": "TextVQA",
        "question_id": item.get("question_id"),
        "image_id": image_id,
        "question": item.get("question", ""),
        "answers": item.get("answers", []),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def metadata_sha(row: dict[str, str]) -> str:
    clean = {k: v for k, v in row.items() if not k.startswith("_")}
    return hashlib.sha256(
        json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--val-json", type=Path, required=True)
    parser.add_argument("--openimages-v3-tar", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--train-count", type=int, default=80)
    parser.add_argument("--val-count", type=int, default=40)
    args = parser.parse_args()

    train = load_questions(args.train_json, "train")
    val = load_questions(args.val_json, "val")
    wanted = set(train) | set(val)
    oi_rows = load_openimages_rows(args.openimages_v3_tar, wanted)

    def eligible_ids(grouped: dict[str, list[dict[str, object]]]) -> list[str]:
        ids = []
        for image_id in grouped:
            meta = oi_rows.get(image_id)
            if not meta:
                continue
            if not is_allowed_metadata_license(meta.get("License", "")):
                continue
            if not meta.get("OriginalLandingURL") or not meta.get("Author"):
                continue
            ids.append(image_id)
        return sorted(ids)

    train_ids = eligible_ids(train)
    val_ids = [i for i in eligible_ids(val) if i not in set(train_ids)]
    rng = random.Random(args.seed)
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    selected: list[tuple[str, str, dict[str, object]]] = []
    for source_split, ids, limit in (("train", train_ids, args.train_count), ("val", val_ids, args.val_count)):
        grouped = train if source_split == "train" else val
        selected_ids = ids[:limit]
        for image_id in selected_ids:
            qa_rows = sorted(grouped[image_id], key=lambda r: int(r.get("question_id", 0)))
            qa = qa_rows[rng.randrange(len(qa_rows))]
            selected.append(("calibration" if source_split == "train" else "heldout", source_split, qa))

    columns = [
        "dataset", "sample_role", "source_split", "question_id", "image_id", "question", "answers_json",
        "image_sha256", "annotation_license_url", "media_source_url", "media_license_name", "media_license_url",
        "permission_evidence", "attribution", "oi_metadata_member", "oi_metadata_row_sha256", "oi_original_md5_base64",
        "oi_rotation_degrees", "question_record_sha256", "prompt_sha256", "eligible_for_local_research",
        "redistributable", "status", "source_page_checked_at", "reviewer_notes"
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for role, source_split, qa in selected:
            image_id = str(qa["_normalized_image_id"])
            meta = oi_rows[image_id]
            license_url = meta.get("License", "")
            author = meta.get("Author", "").strip()
            landing = meta.get("OriginalLandingURL", "").strip()
            attribution = f"{author}; {landing}; CC BY 2.0 (Open Images V3 metadata)"
            question = str(qa.get("question", ""))
            prompt_payload = f"Answer the question using the image.\nQuestion: {question}"
            writer.writerow({
                "dataset": "TextVQA",
                "sample_role": role,
                "source_split": source_split,
                "question_id": qa.get("question_id", ""),
                "image_id": image_id,
                "question": question,
                "answers_json": json.dumps(qa.get("answers", []), ensure_ascii=False, separators=(",", ":")),
                "image_sha256": "",
                "annotation_license_url": TEXTVQA_LICENSE,
                "media_source_url": landing,
                "media_license_name": "Creative Commons Attribution 2.0 (publisher metadata; pending source-page confirmation)",
                "media_license_url": license_url,
                "permission_evidence": f"{OI_V3_README} ; {TEXTVQA_PAGE}",
                "attribution": attribution,
                "oi_metadata_member": f"2017_11/{meta.get('_metadata_split','')}/images.csv",
                "oi_metadata_row_sha256": metadata_sha(meta),
                "oi_original_md5_base64": meta.get("OriginalMD5", ""),
                "oi_rotation_degrees": "",
                "question_record_sha256": canonical_question_sha(qa, image_id),
                "prompt_sha256": hashlib.sha256(prompt_payload.encode("utf-8")).hexdigest(),
                "eligible_for_local_research": "pending_source_page_confirmation",
                "redistributable": "pending_source_page_confirmation",
                "status": "candidate_license_metadata_matched_not_cleared",
                "source_page_checked_at": "",
                "reviewer_notes": "Open Images V3 warns that its per-image license metadata is not warranted; confirm original landing page and rotation before use."
            })

    summary = {
        "schema_version": 1,
        "dataset": "TextVQA",
        "seed": args.seed,
        "requested_counts": {"train_calibration": args.train_count, "val_heldout": args.val_count},
        "selected_counts": {
            "train_calibration": sum(1 for role, split, _ in selected if split == "train"),
            "val_heldout": sum(1 for role, split, _ in selected if split == "val"),
            "total": len(selected)
        },
        "source_question_counts": {"train": sum(map(len, train.values())), "val": sum(map(len, val.values()))},
        "unique_question_images": {"train": len(train), "val": len(val)},
        "openimages_metadata_matches": len(oi_rows),
        "license_candidate_images": {"train": len(train_ids), "val": len(val_ids)},
        "source_sha256": {
            "train_json": file_sha256(args.train_json),
            "val_json": file_sha256(args.val_json),
            "openimages_v3_tar": file_sha256(args.openimages_v3_tar)
        },
        "metadata_license_url_selected": OPENIMAGES_CC_BY_2,
        "source_page_confirmation_count": 0,
        "eligible_for_local_research_count": 0,
        "images_downloaded": 0,
        "status": "candidate_manifest_only_pending_original_page_and_rotation_checks"
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
