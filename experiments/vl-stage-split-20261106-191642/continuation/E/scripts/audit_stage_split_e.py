#!/usr/bin/env python3
"""Audit that E S mode scheduled image E without its LM prefill beside text decode."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {error}") from error
            if isinstance(row, dict):
                rows.append(row)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-jsonl", type=Path, required=True)
    parser.add_argument("--scheduler-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    clients = read_jsonl(args.client_jsonl)
    image_row = next((row for row in clients if row.get("kind") == "image"), None)
    if image_row is None or not image_row.get("response_id"):
        raise SystemExit("client JSONL does not contain a successful image response_id")
    image_id = str(image_row["response_id"])
    scheduler_rows = read_jsonl(args.scheduler_jsonl)
    candidates = []
    for row_index, row in enumerate(scheduler_rows):
        if row.get("kind") != "scheduler_step":
            continue
        enc = row.get("scheduled_encoder_inputs", {}) or {}
        token_schedule = row.get("num_scheduled_tokens", {}) or {}
        matching_ids = [str(req_id) for req_id in enc if str(req_id).startswith(image_id)]
        if not matching_ids:
            continue
        other_decode = {
            str(req_id): int(count)
            for req_id, count in token_schedule.items()
            if int(count) > 0 and not str(req_id).startswith(image_id)
        }
        if all(req_id not in token_schedule or int(token_schedule[req_id]) == 0 for req_id in matching_ids) and other_decode:
            candidates.append({
                "scheduler_row_index": row_index,
                "image_internal_request_ids": matching_ids,
                "image_encoder_indices": {req_id: enc[req_id] for req_id in matching_ids},
                "image_has_lm_tokens_in_step": False,
                "other_scheduled_decode_tokens": other_decode,
                "total_num_scheduled_tokens": row.get("total_num_scheduled_tokens"),
            })

    result = {
        "schema_version": 1,
        "valid": bool(candidates),
        "image_response_id": image_id,
        "client_jsonl_sha256": sha256(args.client_jsonl),
        "scheduler_jsonl_sha256": sha256(args.scheduler_jsonl),
        "candidate_split_steps": candidates,
        "limitations": [
            "This checks only a synthetic mechanism smoke; it is not a latency, quality, or formal-load result.",
            "It proves the scheduler emitted image encoder work without image LM tokens while text tokens were scheduled in the same step; it does not validate GeoPrune or token compaction.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if not result["valid"]:
        raise SystemExit("no S split scheduler step found")


if __name__ == "__main__":
    main()
