#!/usr/bin/env python3
"""Submit every page of a fixed DocVQA-2026 document to the local E vLLM server."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8285/v1")
    parser.add_argument("--model", default="qwen3vl4b")
    parser.add_argument("--edge", type=int, choices=(512, 768), required=True)
    parser.add_argument("--doc-id", default="business_report_1")
    parser.add_argument(
        "--sample-id",
        default="business_report_1:business_report_1_q1",
    )
    parser.add_argument("--documents-jsonl", type=Path, required=True)
    parser.add_argument("--candidates-jsonl", type=Path, required=True)
    parser.add_argument("--context-audit-json", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()

    documents = [
        json.loads(line)
        for line in args.documents_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    document = next((row for row in documents if row["doc_id"] == args.doc_id), None)
    if document is None:
        raise ValueError(f"unknown DocVQA document: {args.doc_id}")
    candidates = [
        json.loads(line)
        for line in args.candidates_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidate = next(
        (row for row in candidates if row["sample_id"] == args.sample_id),
        None,
    )
    if candidate is None or candidate["doc_id"] != args.doc_id:
        raise ValueError(f"question is not a candidate in document {args.doc_id}: {args.sample_id}")
    context_audit = json.loads(args.context_audit_json.read_text(encoding="utf-8"))
    audited_doc = next(row for row in context_audit["documents"] if row["doc_id"] == args.doc_id)
    visual_tokens = audited_doc["per_edge"][str(args.edge)]["total_visual_tokens"]
    visual_kv_gib = audited_doc["per_edge"][str(args.edge)]["visual_only_kv_gib"]

    content: list[dict[str, Any]] = []
    image_records: list[dict[str, Any]] = []
    for page in document["pages"]:
        image_path = args.images_root / page["relative_path"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": image_path.as_uri()},
        })
        image_records.append({
            "page_index": page["page_index"],
            "path": str(image_path),
            "sha256_png": page["sha256_png"],
            "image_url": image_path.as_uri(),
        })
    instruction = (
        "The images above are every page of one document, in original page order. "
        "Use the document to answer the question. Reply with one line in the format "
        "FINAL ANSWER: <answer>. Do not add an explanation. "
        f"Question: {candidate['question']}"
    )
    content.append({"type": "text", "text": instruction})
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "stream": False,
        "mm_processor_kwargs": {
            "min_pixels": args.edge * args.edge,
            "max_pixels": args.edge * args.edge,
        },
    }
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic_ns()
    status: int | None = None
    response_body: bytes = b""
    error: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            status = response.status
            response_body = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        response_body = exc.read()
        error = f"HTTPError: {exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        response_body = str(exc).encode("utf-8", errors="replace")
    elapsed_ms = (time.monotonic_ns() - started) / 1e6
    parsed: dict[str, Any] | None = None
    try:
        parsed = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    choice = (parsed or {}).get("choices", [{}])[0]
    message = choice.get("message", {})
    output_text = message.get("content") if isinstance(message, dict) else None
    result = {
        "smoke_only": True,
        "quality_score": None,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "VLR-CVC/DocVQA-2026",
        "split": "public val",
        "doc_id": args.doc_id,
        "sample_id": args.sample_id,
        "doc_category": document["doc_category"],
        "page_count": len(document["pages"]),
        "processor_edge": args.edge,
        "expected_visual_tokens_from_cpu_audit": visual_tokens,
        "expected_visual_only_kv_gib_from_cpu_audit": visual_kv_gib,
        "images": image_records,
        "question": candidate["question"],
        "request_endpoint": endpoint,
        "request_sha256": sha256_bytes(payload),
        "request": body,
        "http_status": status,
        "elapsed_client_ms": elapsed_ms,
        "error": error,
        "response_json": parsed,
        "response_body_text": response_body.decode("utf-8", errors="replace"),
        "response_id": (parsed or {}).get("id"),
        "usage": (parsed or {}).get("usage"),
        "finish_reason": choice.get("finish_reason"),
        "answer_text": output_text,
        "transport_response_received": status == 200 and parsed is not None and error is None,
        "generation_stopped_normally": choice.get("finish_reason") == "stop",
        "response_complete": status == 200 and bool(output_text) and error is None and choice.get("finish_reason") == "stop",
    }
    atomic_json(args.output, result)
    print(json.dumps({
        "http_status": status,
        "elapsed_client_ms": round(elapsed_ms, 2),
        "doc_id": args.doc_id,
        "page_count": len(document["pages"]),
        "processor_edge": args.edge,
        "expected_visual_tokens": visual_tokens,
        "finish_reason": result["finish_reason"],
        "response_complete": result["response_complete"],
        "output": str(args.output),
        "error": error,
    }, ensure_ascii=False))
    return 0 if result["response_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
