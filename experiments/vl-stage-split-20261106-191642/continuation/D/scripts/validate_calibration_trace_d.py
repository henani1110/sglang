#!/usr/bin/env python3
"""Annotate calibration attempts that have a complete chat first-token trace join."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
    return rows


def request_steps(rows: list[dict[str, Any]], response_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    encoder = None
    prefill = None
    for row in rows:
        if row.get("kind") != "scheduler_step":
            continue
        encoder_ids = row.get("scheduled_encoder_inputs", {})
        token_ids = row.get("num_scheduled_tokens", {})
        if encoder is None and any(str(key).startswith(response_id) for key in encoder_ids):
            encoder = row
        if prefill is None and any(
            str(key).startswith(response_id) and int(value) > 0
            for key, value in token_ids.items()
        ):
            prefill = row
        if encoder is not None and prefill is not None:
            break
    return encoder, prefill


def validate_row(run: dict[str, Any]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    if run.get("status") != "completed" or int(run.get("process_exit", 1)) != 0:
        errors.append({"reason": "campaign_not_completed", "status": run.get("status"), "process_exit": run.get("process_exit")})
    paths: dict[str, Path] = {}
    for field in ("client_events_path", "scheduler_events_path", "server_events_path"):
        path = Path(str(run.get(field, "")))
        if not path.is_file() or path.stat().st_size == 0:
            errors.append({"reason": "event_file_missing", "field": field, "path": str(path)})
        else:
            paths[field] = path
    if errors:
        return {"analysis_eligible": False, "image_count": 0, "matched_images": 0, "errors": errors}

    client = read_jsonl(paths["client_events_path"])
    scheduler = read_jsonl(paths["scheduler_events_path"])
    server = read_jsonl(paths["server_events_path"])
    images = [
        row for row in client
        if row.get("kind") == "image_response" and row.get("phase") == "measurement"
    ]
    if not images:
        errors.append({"reason": "no_measurement_image_responses"})
    matched = 0
    for image in images:
        request_id = str(image.get("request_id", ""))
        if image.get("error") or not request_id or not image.get("first_token_ns"):
            errors.append({"reason": "image_response_incomplete", "request_id": request_id, "error": image.get("error")})
            continue
        encoder, prefill = request_steps(scheduler, request_id)
        if encoder is None or prefill is None:
            errors.append({"reason": "scheduler_request_unmatched", "request_id": request_id})
            continue
        engine = [
            row for row in server
            if row.get("kind") == "engine_result_ready_to_api"
            and str(row.get("request_id", "")) == request_id
            and int(row.get("token_index_start", -1)) == 0
        ]
        api = [
            row for row in server
            if row.get("kind") == "api_yield"
            and str(row.get("request_id", "")) == request_id
            and int(row.get("token_index_start", -1)) == 0
        ]
        if not engine:
            errors.append({"reason": "engine_first_token_unmatched", "request_id": request_id})
            continue
        if not api:
            errors.append({"reason": "api_first_token_unmatched", "request_id": request_id})
            continue
        engine_first = min(engine, key=lambda row: int(row["time_ns"]))
        api_first = min(api, key=lambda row: int(row["time_ns"]))
        client_tokens = [int(value) for value in image.get("token_ids", [])]
        engine_tokens = [int(value) for value in engine_first.get("token_ids", [])]
        api_tokens = [int(value) for value in api_first.get("token_ids", [])]
        if not client_tokens or not engine_tokens or not api_tokens:
            errors.append({"reason": "first_token_ids_missing", "request_id": request_id})
            continue
        if client_tokens[0] != engine_tokens[0] or client_tokens[0] != api_tokens[0]:
            errors.append({
                "reason": "first_token_id_mismatch",
                "request_id": request_id,
                "client": client_tokens[0],
                "engine": engine_tokens[0],
                "api": api_tokens[0],
            })
            continue
        times = (int(engine_first["time_ns"]), int(api_first["time_ns"]), int(image["first_token_ns"]))
        if not times[0] <= times[1] <= times[2]:
            errors.append({"reason": "first_token_event_order_invalid", "request_id": request_id, "times_ns": list(times)})
            continue
        matched += 1
    return {
        "analysis_eligible": not errors and bool(images) and matched == len(images),
        "image_count": len(images),
        "matched_images": matched,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--audit-log", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.index.read_text(encoding="utf-8")) if args.index.is_file() else {"schema_version": 1, "runs": []}
    changes: list[dict[str, Any]] = []
    eligible = 0
    for row in payload.get("runs", []):
        qa = validate_row(row)
        if row.get("analysis_eligible") != qa["analysis_eligible"] or row.get("trace_qa") != qa:
            row["analysis_eligible"] = qa["analysis_eligible"]
            row["trace_qa"] = qa
            changes.append({
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "event": "trace_eligibility_updated",
                "run_id": row.get("run_id"),
                **qa,
            })
        eligible += int(qa["analysis_eligible"])
    temp = args.index.with_suffix(args.index.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, args.index)
    args.audit_log.parent.mkdir(parents=True, exist_ok=True)
    with args.audit_log.open("a", encoding="utf-8") as handle:
        for row in changes:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"indexed_runs": len(payload.get("runs", [])), "analysis_eligible_runs": eligible, "updates": len(changes)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
