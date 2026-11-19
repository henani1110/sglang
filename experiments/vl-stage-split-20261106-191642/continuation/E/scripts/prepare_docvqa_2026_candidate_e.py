#!/usr/bin/env python3
"""Download and fingerprint only the public DocVQA-2026 validation split."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ID = "VLR-CVC/DocVQA-2026"
CARD_URL = "https://huggingface.co/datasets/VLR-CVC/DocVQA-2026"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def safe_name(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return cleaned or "unknown"


def nested_values(group: Any, key: str) -> list[Any]:
    if isinstance(group, dict):
        value = group.get(key)
        return list(value) if isinstance(value, (list, tuple)) else []
    if isinstance(group, (list, tuple)):
        return [item.get(key) for item in group if isinstance(item, dict)]
    return []


def jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--revision", help="Pinned Hugging Face dataset commit SHA; defaults to current repo SHA.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(args.cache_dir / "hf-home")
    os.environ["HF_HUB_CACHE"] = str(args.cache_dir / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(args.cache_dir / "datasets")

    from datasets import Image as DatasetImage
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from PIL import Image as PILImage

    del DatasetImage  # Imported to make the expected datasets image feature explicit.
    PILImage.MAX_IMAGE_PIXELS = None

    api = HfApi()
    info = api.dataset_info(REPO_ID, revision=args.revision)
    revision = args.revision or info.sha
    card_data = info.card_data.to_dict() if info.card_data is not None else {}
    declared_license = card_data.get("license") if isinstance(card_data, dict) else None
    dataset = load_dataset(
        REPO_ID,
        split="val",
        revision=revision,
        cache_dir=str(args.cache_dir / "datasets"),
    )

    document_rows: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []
    category_docs: Counter[str] = Counter()
    category_questions: Counter[str] = Counter()
    total_pages = 0

    for row_index, record in enumerate(dataset):
        doc_id = str(record["doc_id"])
        category = str(record.get("doc_category", "unknown"))
        category_docs[category] += 1
        safe_doc = safe_name(doc_id)
        pages = record.get("document") or []
        if not isinstance(pages, (list, tuple)):
            pages = [pages]
        page_rows: list[dict[str, Any]] = []
        for page_index, page in enumerate(pages):
            if isinstance(page, dict) and page.get("bytes"):
                image = PILImage.open(__import__("io").BytesIO(page["bytes"]))
            elif isinstance(page, dict) and page.get("path"):
                image = PILImage.open(page["path"])
            else:
                image = page
            image = image.convert("RGB")
            buffer = __import__("io").BytesIO()
            image.save(buffer, format="PNG", optimize=False)
            png_bytes = buffer.getvalue()
            digest = sha256_bytes(png_bytes)
            relative = Path("images") / safe_doc / f"page-{page_index + 1:03d}-{digest[:12]}.png"
            target = args.output_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(png_bytes)
            page_rows.append({
                "page_index": page_index,
                "relative_path": relative.as_posix(),
                "sha256_png": digest,
                "width": image.width,
                "height": image.height,
                "bytes": len(png_bytes),
            })

        questions = nested_values(record.get("questions"), "question")
        question_ids = nested_values(record.get("questions"), "question_id")
        answer_ids = nested_values(record.get("answers"), "question_id")
        answers = nested_values(record.get("answers"), "answer")
        if len(questions) != len(question_ids) or len(answer_ids) != len(answers):
            raise ValueError(f"question/answer arrays do not align for doc_id={doc_id}")
        answer_map = {str(qid): answer for qid, answer in zip(answer_ids, answers)}
        if len(set(map(str, question_ids))) != len(question_ids):
            raise ValueError(f"duplicate question_id in doc_id={doc_id}")
        if not set(map(str, question_ids)).issubset(answer_map):
            raise ValueError(f"missing answer ids for doc_id={doc_id}")

        document_rows.append({
            "doc_id": doc_id,
            "doc_category": category,
            "dataset_row_index": row_index,
            "page_count": len(page_rows),
            "pages": page_rows,
            "question_count": len(questions),
            "source_repo": REPO_ID,
            "source_revision": revision,
            "source_split": "val",
            "rights_status": "public_validation_local_experiment_only_license_scope_pending",
        })
        total_pages += len(page_rows)
        category_questions[category] += len(questions)
        for question_id, question in zip(question_ids, questions):
            qid = str(question_id)
            question_rows.append({
                "sample_id": qid,
                "doc_id": doc_id,
                "doc_category": category,
                "question": question,
                "answer": answer_map[qid],
                "page_count": len(page_rows),
                "pages": page_rows,
                "source_repo": REPO_ID,
                "source_revision": revision,
                "source_split": "val",
                "dataset_card_url": CARD_URL,
                "rights_status": "public_validation_local_experiment_only_license_scope_pending",
            })

    jsonl_write(args.output_dir / "documents.jsonl", document_rows)
    jsonl_write(args.output_dir / "questions.jsonl", question_rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": REPO_ID,
        "source_revision": revision,
        "source_split": "val",
        "dataset_card_url": CARD_URL,
        "dataset_card_declared_license": declared_license,
        "rights_status": "public_validation_local_experiment_only_license_scope_pending",
        "document_count": len(document_rows),
        "question_count": len(question_rows),
        "page_count": total_pages,
        "documents_by_category": dict(sorted(category_docs.items())),
        "questions_by_category": dict(sorted(category_questions.items())),
        "multi_page_document_count": sum(row["page_count"] > 1 for row in document_rows),
        "manifest_files": ["documents.jsonl", "questions.jsonl"],
        "limitations": [
            "Only the public validation split is loaded; the test split is never requested.",
            "The Hugging Face card does not declare a dataset license; local evaluation scope is recorded, but full rights clearance is not claimed.",
            "Image hashes cover the exported RGB PNG bytes, not the encoded bytes originally stored in Parquet.",
            "Question rows are clustered by doc_id; calibration and heldout splits must be document-disjoint.",
            "Do not redistribute source images or claim that this supplement is interchangeable with DocVQA 2020-23.",
        ],
    }
    (args.output_dir / "source-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "document_count": summary["document_count"],
        "question_count": summary["question_count"],
        "page_count": total_pages,
        "multi_page_document_count": summary["multi_page_document_count"],
        "source_revision": revision,
        "declared_license": declared_license,
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
