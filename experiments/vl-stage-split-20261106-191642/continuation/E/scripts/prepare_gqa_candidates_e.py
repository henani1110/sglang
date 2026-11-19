#!/usr/bin/env python3
"""Build a rights-pending GQA candidate manifest from official metadata only.

No benchmark image is downloaded. GQA image IDs are joined through Visual
Genome image metadata to COCO train/val image records, then filtered by the
COCO per-image license metadata. This is a candidate filter, not a rights
determination for either the GQA Q/A annotations or the third-party image.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


DEFAULT_PROMPT = "Answer the image question concisely. "
FIELDS = (
    "dataset", "sample_role", "source_split", "question_id", "image_id", "question",
    "answers_json", "image_sha256", "annotation_license_url", "media_source_url",
    "media_license_name", "media_license_url", "permission_evidence", "attribution",
    "coco_file_name", "coco_url", "coco_license_id", "coco_image_record_sha256",
    "question_record_sha256", "prompt_sha256", "eligible_for_local_research",
    "redistributable", "status", "source_page_checked_at", "reviewer_notes",
    "gqa_image_id", "visual_genome_image_id", "visual_genome_record_sha256",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(row: dict[str, Any]) -> str:
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


class TopLevelJsonObject:
    """Stream a JSON object whose values are small, avoiding a full-file load."""

    def __init__(self, path: Path, chunk_size: int = 1024 * 1024) -> None:
        self.stream = path.open("r", encoding="utf-8")
        self.chunk_size = chunk_size
        self.buffer = ""
        self.eof = False
        self.decoder = json.JSONDecoder()

    def close(self) -> None:
        self.stream.close()

    def _fill(self) -> bool:
        if self.eof:
            return False
        chunk = self.stream.read(self.chunk_size)
        if not chunk:
            self.eof = True
            return False
        self.buffer += chunk
        return True

    def _skip_ws(self) -> None:
        while True:
            stripped = self.buffer.lstrip()
            self.buffer = stripped
            if self.buffer or not self._fill():
                return

    def _expect(self, char: str) -> None:
        self._skip_ws()
        while not self.buffer:
            if not self._fill():
                raise ValueError(f"unexpected EOF; expected {char!r}")
        if self.buffer[0] != char:
            raise ValueError(f"expected {char!r}, found {self.buffer[0]!r}")
        self.buffer = self.buffer[1:]

    def _decode_value(self) -> Any:
        self._skip_ws()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer)
                self.buffer = self.buffer[end:]
                return value
            except json.JSONDecodeError:
                if not self._fill():
                    raise

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        self._expect("{")
        first = True
        while True:
            self._skip_ws()
            if not self.buffer and not self._fill():
                raise ValueError("unexpected EOF in top-level object")
            if self.buffer[0] == "}":
                self.buffer = self.buffer[1:]
                return
            if not first:
                self._expect(",")
            key = self._decode_value()
            if not isinstance(key, str):
                raise ValueError("top-level object key must be a string")
            self._expect(":")
            yield key, self._decode_value()
            first = False


def load_coco(path: Path) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    licenses = {int(row["id"]): row for row in data.get("licenses", [])}
    images = {int(row["id"]): row for row in data.get("images", [])}
    if not licenses or not images:
        raise ValueError(f"invalid COCO instances metadata: {path}")
    return licenses, images


def load_vg_coco_map(path: Path) -> tuple[dict[int, tuple[int, dict[str, Any]]], int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[int, tuple[int, dict[str, Any]]] = {}
    duplicate_coco = 0
    seen_coco: set[int] = set()
    for row in data:
        coco_id = row.get("coco_id")
        if coco_id in (None, "", 0, "0"):
            continue
        try:
            vg_id = int(row["image_id"])
            coco_id_int = int(coco_id)
        except (KeyError, TypeError, ValueError):
            continue
        if coco_id_int in seen_coco:
            duplicate_coco += 1
        seen_coco.add(coco_id_int)
        mapping[vg_id] = (coco_id_int, row)
    return mapping, duplicate_coco


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gqa-questions", type=Path, required=True, help="official val_balanced_questions.json")
    parser.add_argument("--visualgenome-image-data", type=Path, required=True)
    parser.add_argument("--coco-instances", type=Path, nargs="+", required=True, help="instances_train2014.json and/or val2014.json")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--license-ids", type=int, nargs="+", default=[4, 5])
    args = parser.parse_args()

    licenses: dict[int, dict[str, Any]] = {}
    coco_images: dict[int, dict[str, Any]] = {}
    for path in args.coco_instances:
        this_licenses, this_images = load_coco(path)
        if licenses and licenses != this_licenses:
            raise ValueError("COCO license maps differ between split files")
        licenses = this_licenses
        overlap = set(coco_images).intersection(this_images)
        if overlap:
            raise ValueError(f"overlapping COCO image IDs across splits: {sorted(overlap)[:5]}")
        coco_images.update(this_images)
    vg_map, duplicate_coco = load_vg_coco_map(args.visualgenome_image_data)

    question_file_hash = sha256_file(args.gqa_questions)
    vg_file_hash = sha256_file(args.visualgenome_image_data)
    coco_file_hashes = {str(path): sha256_file(path) for path in args.coco_instances}
    eligible_media_ids = {
        image_id for image_id, row in coco_images.items()
        if int(row.get("license", -1)) in args.license_ids
    }

    # Keep one deterministic pseudo-random question per distinct COCO image,
    # then choose images via a separately salted stable hash priority.
    best_by_coco: dict[int, tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    total_questions = 0
    missing_vg = 0
    missing_coco = 0
    noncandidate_license = 0
    stream = TopLevelJsonObject(args.gqa_questions)
    try:
        for question_id, value in stream:
            total_questions += 1
            if not isinstance(value, dict):
                continue
            raw_vg_id = value.get("imageId")
            try:
                vg_id = int(raw_vg_id)
            except (TypeError, ValueError):
                missing_vg += 1
                continue
            vg_match = vg_map.get(vg_id)
            if vg_match is None:
                missing_vg += 1
                continue
            coco_id, vg_row = vg_match
            coco_row = coco_images.get(coco_id)
            if coco_row is None:
                missing_coco += 1
                continue
            if coco_id not in eligible_media_ids:
                noncandidate_license += 1
                continue
            question_record = {"question_id": str(question_id), **value}
            question_priority = sha256_bytes(f"{args.seed}:gqa-question:{question_id}".encode("utf-8"))
            existing = best_by_coco.get(coco_id)
            if existing is None or question_priority < existing[0]:
                best_by_coco[coco_id] = (question_priority, str(question_id), question_record, vg_row, coco_row)
    finally:
        stream.close()

    image_order = sorted(
        best_by_coco,
        key=lambda image_id: (sha256_bytes(f"{args.seed}:gqa-image:{image_id}".encode("utf-8")), image_id),
    )
    selected_ids = image_order[: args.count]
    prompt = DEFAULT_PROMPT
    evidence = (
        "GQA official download page https://cs.stanford.edu/people/dorarad/gqa/download.html does not state an overall Q/A data license; "
        "GQA official homepage https://cs.stanford.edu/people/dorarad/gqa/index.html describes images as COCO/Flickr sourced. "
        "Image candidate was joined through Visual Genome author metadata https://homes.cs.washington.edu/~ranjay/visualgenome/api.html "
        "to COCO per-image license metadata; see COCO Terms https://cocodataset.org/dataset/termsofuse.htm. "
        "Q/A annotation permission and each original image source page remain unverified."
    )
    output_rows: list[dict[str, Any]] = []
    license_counts: Counter[str] = Counter()
    for coco_id in selected_ids:
        _, question_id, question_record, vg_row, coco_row = best_by_coco[coco_id]
        license_id = int(coco_row.get("license", -1))
        license_row = licenses.get(license_id, {})
        question = str(question_record.get("question", ""))
        answer = question_record.get("answer", "")
        license_counts[str(license_id)] += 1
        output_rows.append({
            "dataset": "GQA",
            "sample_role": "quality_candidate",
            "source_split": "val_balanced_questions",
            "question_id": question_id,
            "image_id": str(coco_id),
            "question": question,
            "answers_json": json.dumps([str(answer)], ensure_ascii=False),
            "image_sha256": "",
            "annotation_license_url": "",
            "media_source_url": str(coco_row.get("flickr_url", "")),
            "media_license_name": str(license_row.get("name", "")),
            "media_license_url": str(license_row.get("url", "")),
            "permission_evidence": evidence,
            "attribution": "pending original source-page confirmation",
            "coco_file_name": str(coco_row.get("file_name", "")),
            "coco_url": str(coco_row.get("coco_url", "")),
            "coco_license_id": license_id,
            "coco_image_record_sha256": canonical_hash(coco_row),
            "question_record_sha256": canonical_hash(question_record),
            "prompt_sha256": sha256_bytes((prompt + question).encode("utf-8")),
            "eligible_for_local_research": "pending_annotation_and_source_page_review",
            "redistributable": "pending_annotation_and_source_page_review",
            "status": "candidate_media_license_metadata_matched_annotation_scope_pending",
            "source_page_checked_at": "",
            "reviewer_notes": (
                f"GQA imageId={question_record.get('imageId')}; Visual Genome image_id={vg_row.get('image_id')}; "
                f"COCO image_id={coco_id}, license_id={license_id}. No image downloaded. "
                "GQA annotation-data license not identified on official download page; verify Q/A permission and original Flickr/source page, "
                "author/attribution, current media terms, and compliant retrieval path before eligibility."
            ),
            "gqa_image_id": str(question_record.get("imageId", "")),
            "visual_genome_image_id": str(vg_row.get("image_id", "")),
            "visual_genome_record_sha256": canonical_hash(vg_row),
        })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream_out:
        writer = csv.DictWriter(stream_out, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    source_hashes = {
        "gqa_val_balanced_questions": question_file_hash,
        "visualgenome_image_data_json": vg_file_hash,
        "coco_instances": coco_file_hashes,
    }
    summary = {
        "schema_version": 1,
        "status": "candidate_manifest_created_pending_annotation_and_source_page_review",
        "seed": args.seed,
        "target_candidate_count": args.count,
        "license_ids_selected": args.license_ids,
        "gqa_annotation_license_scope_confirmed": False,
        "gqa_questions_scanned": total_questions,
        "questions_missing_visualgenome_mapping": missing_vg,
        "questions_missing_coco_mapping": missing_coco,
        "question_rows_with_non_candidate_coco_license": noncandidate_license,
        "unique_metadata_candidate_images": len(best_by_coco),
        "candidate_rows": len(output_rows),
        "candidate_unique_coco_image_ids": len({row["image_id"] for row in output_rows}),
        "selected_license_id_counts": dict(sorted(license_counts.items())),
        "visualgenome_duplicate_coco_id_records": duplicate_coco,
        "source_sha256": source_hashes,
        "candidate_manifest_sha256": sha256_file(args.output_csv),
        "source_page_confirmed": 0,
        "annotation_permission_confirmed": 0,
        "eligible_for_local_research": 0,
        "images_downloaded": 0,
        "interpretation": (
            "COCO license metadata was used only as a candidate filter. It does not establish GQA annotation rights or current rights for any image. "
            "No candidate is eligible and no image was downloaded until the annotation license and each original image source page are verified."
        ),
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "gqa_questions_scanned": total_questions,
        "unique_metadata_candidate_images": len(best_by_coco),
        "candidate_rows": len(output_rows),
        "candidate_manifest": str(args.output_csv),
        "summary": str(args.output_summary),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
