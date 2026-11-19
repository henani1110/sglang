#!/usr/bin/env python3
"""Analyze B-stage E->D->P ordering and compare exact outputs to native N."""
from __future__ import annotations

import json
import os
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "continuation" / "raw"
EXTRACTED = RAW / "b_cells"
EVIDENCE_ROOT = EXTRACTED / "b_evidence"
LABELS = ("B-N-r0-retry1", "B-S-r0", "B-S-r1-2")
BASELINE_LABEL = LABELS[0]
CANCEL_LABEL = "B-S-cancel-r0"


def ensure_extracted() -> None:
    archive = RAW / "B-evidence.tar.gz"
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        root = EXTRACTED.resolve()
        for member in tar.getmembers():
            target = (EXTRACTED / member.name).resolve()
            if os.path.commonpath((str(root), str(target))) != str(root):
                raise ValueError(f"archive member escapes destination: {member.name}")
            tar.extract(member, EXTRACTED)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key_matches(key: str, request_id: str | None) -> bool:
    return bool(request_id and key.startswith(request_id + "-"))


def load_cell(label: str, baseline_episode: dict[str, Any] | None = None) -> dict[str, Any]:
    root = EVIDENCE_ROOT
    result = json.loads((root / "results" / f"{label}.json").read_text())
    clients = read_jsonl(root / "logs" / f"{label}-client-events.jsonl")
    scheduler = read_jsonl(root / "logs" / f"{label}-scheduler-events.jsonl")
    server = read_jsonl(root / "logs" / f"{label}-server-events.jsonl")
    meta_path = root / "logs" / f"{label}-server-events.jsonl.async-meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    summaries = []
    for episode in result.get("episodes", []):
        eid = episode["episode_id"]
        image = episode.get("image") or {}
        image_id = image.get("request_id")
        text_ids = [row.get("request_id") for row in episode.get("text", [])]
        client_rows = [row for row in clients if row.get("episode_id") == eid]
        anchor_row = next((row for row in client_rows if row.get("kind") == "episode_anchor"), {})
        anchor_ns = int(anchor_row.get("time_ns", episode.get("anchor_ns", 0)))
        first_image_token_ns = min(
            (int(row["time_ns"]) for row in client_rows
             if row.get("kind") == "sse_chunk" and row.get("lane") is None and row.get("token_ids")),
            default=None,
        )
        split_rows = []
        p_rows = []
        for row in scheduler:
            mapping = row.get("num_scheduled_tokens") or {}
            enc = row.get("scheduled_encoder_inputs") or {}
            image_encoder_keys = [key for key in enc if key_matches(key, image_id)]
            image_token_keys = [key for key in mapping if key_matches(key, image_id)]
            if image_encoder_keys and not image_token_keys:
                text_map = {lane: value for lane, value in mapping.items()
                            if any(key_matches(lane, req_id) for req_id in text_ids)}
                split_rows.append({
                    "time_ns": int(row["time_ns"]),
                    "relative_to_anchor_ms": (int(row["time_ns"]) - anchor_ns) / 1e6,
                    "image_key": image_encoder_keys[0],
                    "encoder_inputs": enc[image_encoder_keys[0]],
                    "text_decode_tokens": text_map,
                    "total_num_scheduled_tokens": row.get("total_num_scheduled_tokens"),
                })
            if image_token_keys:
                p_rows.append({
                    "time_ns": int(row["time_ns"]),
                    "relative_to_anchor_ms": (int(row["time_ns"]) - anchor_ns) / 1e6,
                    "image_key": image_token_keys[0],
                    "image_scheduled_tokens": mapping[image_token_keys[0]],
                    "encoder_inputs": enc.get(image_token_keys[0]),
                    "total_num_scheduled_tokens": row.get("total_num_scheduled_tokens"),
                })
        split = min(split_rows, key=lambda row: row["time_ns"]) if split_rows else None
        later_p = [row for row in p_rows if split and row["time_ns"] > split["time_ns"]]
        first_p = min(later_p, key=lambda row: row["time_ns"]) if later_p else None
        visual = [
            row for row in server
            if row.get("kind") == "visual_forward"
            and row.get("mode") == "R"
            and row.get("image_index") == episode.get("image_index")
            and split is not None
            and first_p is not None
            and int(row.get("time_ns", 0)) >= split["time_ns"]
            and int(row.get("time_ns", 0)) <= first_p["time_ns"]
        ]
        finished_rows = [
            row for row in scheduler
            if any(key_matches(str(req_id), image_id) for req_id in row.get("finished_req_ids", []))
        ]
        expected_text = [row.get("token_ids") for row in episode.get("text", [])]
        expected_image = image.get("token_ids")
        output_equal = None
        if baseline_episode is not None:
            output_equal = (
                expected_text == [row.get("token_ids") for row in baseline_episode.get("text", [])]
                and expected_image == (baseline_episode.get("image") or {}).get("token_ids")
            )
        summaries.append({
            "episode_id": eid,
            "validation_errors": episode.get("validation_errors") or [],
            "text_token_counts": [len(ids or []) for ids in expected_text],
            "image_token_count": len(expected_image or []),
            "image_sha256": image.get("image_sha256"),
            "output_ids_equal_to_native": output_equal,
            "anchor_ns": anchor_ns,
            "first_image_token_ns": first_image_token_ns,
            "split_steps": split_rows,
            "first_P_after_split": first_p,
            "visual_forward_between_split_and_P": visual,
            "image_finished_events": finished_rows,
        })
    event_kinds = Counter(row.get("kind", "unknown") for row in server)
    return {
        "label": label,
        "status": result.get("status"),
        "episodes": summaries,
        "scheduler_event_count": len(scheduler),
        "server_event_count": len(server),
        "server_event_kinds": dict(event_kinds),
        "async_writer": meta,
    }


def analyze_cancellation() -> dict[str, Any]:
    root = EVIDENCE_ROOT
    label = CANCEL_LABEL
    result = json.loads((root / "results" / f"{label}.json").read_text())
    clients = read_jsonl(root / "logs" / f"{label}-client-events.jsonl")
    scheduler = read_jsonl(root / "logs" / f"{label}-scheduler-events.jsonl")
    server = read_jsonl(root / "logs" / f"{label}-server-events.jsonl")
    image_id = result.get("image_request_id")
    request_keys = [key for row in scheduler for key in (row.get("num_scheduled_tokens") or {}) if key_matches(key, image_id)]
    split = next((row for row in scheduler if any(
        key_matches(key, image_id) for key in (row.get("scheduled_encoder_inputs") or {})
    ) and not any(key_matches(key, image_id) for key in (row.get("num_scheduled_tokens") or {}))), None)
    p_rows = [row for row in scheduler if any(key_matches(key, image_id) for key in (row.get("num_scheduled_tokens") or {}))]
    first_p = min(p_rows, key=lambda row: int(row.get("time_ns", 0))) if p_rows else None
    cancel = next((row for row in clients if row.get("kind") == "client_stream_cancel"), None)
    finished = [
        row for row in scheduler
        if any(key_matches(str(req_id), image_id) for req_id in row.get("finished_req_ids", []))
    ]
    finish = min(finished, key=lambda row: int(row.get("time_ns", 0))) if finished else None
    visual = [
        row for row in server
        if row.get("kind") == "visual_forward"
        and row.get("mode") == "R"
        and (split is None or int(row.get("time_ns", 0)) >= int(split["time_ns"]))
        and (first_p is None or int(row.get("time_ns", 0)) <= int(first_p["time_ns"]))
    ]
    meta_path = root / "logs" / f"{label}-server-events.jsonl.async-meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    runner_path = root / "runlogs" / f"{label}-runner-status.txt"
    runner_status = runner_path.read_text() if runner_path.exists() else None
    return {
        "label": label,
        "status": result.get("status"),
        "image_request_id": image_id,
        "partial_image_tokens": result.get("image_partial_tokens"),
        "client_cancel_record": cancel,
        "split_encoder_only_step": split,
        "first_P_step": first_p,
        "visual_forward_between_split_and_P": visual,
        "image_finished_event": finish,
        "finished_after_client_cancel": bool(cancel and finish and int(finish["time_ns"]) >= int(cancel["time_ns"])),
        "text_token_counts": [item.get("token_count") for item in result.get("text", [])],
        "text_errors": [item.get("error") for item in result.get("text", [])],
        "image_schedule_rows": len(request_keys),
        "async_writer": meta,
        "runner_status": runner_status,
    }


def main() -> None:
    ensure_extracted()
    baseline_result = json.loads((EVIDENCE_ROOT / "results" / f"{BASELINE_LABEL}.json").read_text())
    baseline_episode = baseline_result["episodes"][0]
    cells = [load_cell(BASELINE_LABEL)]
    cells += [load_cell(label, baseline_episode) for label in LABELS[1:]]
    cancellation = analyze_cancellation()
    report = {
        "method": {
            "comparison": "same image bytes and frozen four text prompts; exact sampled token-ID comparison to N",
            "split_detection": "image request appears in scheduled_encoder_inputs but not num_scheduled_tokens; text D appears in the same scheduler step; image P is matched on a later step",
            "visual_forward_scope": "mode=R event between encoder-only step and first P step; warmup events are excluded",
        },
        "cells": cells,
        "cancellation": cancellation,
    }
    out_json = ROOT / "results" / "B-semantic-analysis.json"
    out_md = ROOT / "results" / "B-semantic-analysis.md"
    out_json.parent.mkdir(exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# B 阶段 E→D→P 正确性复核", "",
             "|单元|episode|错误|文本 token 数|图文 token 数|与 N token IDs 一致|split D 数|后续 P token 数|E 次数|", "|---|---|---:|---|---:|---|---:|---:|---:|"]
    for cell in cells:
        for episode in cell["episodes"]:
            split = episode["split_steps"]
            dcount = len((split[0]["text_decode_tokens"] if split else {}))
            p = episode["first_P_after_split"]
            lines.append(
                f"|{cell['label']}|{episode['episode_id']}|{len(episode['validation_errors'])}|"
                f"{episode['text_token_counts']}|{episode['image_token_count']}|"
                f"{episode['output_ids_equal_to_native']}|{dcount}|"
                f"{p['image_scheduled_tokens'] if p else 'N/A'}|"
                f"{len(episode['visual_forward_between_split_and_P'])}|"
            )
    lines += ["", "## 判读", "",
              "S 的正确性只在出现真实 encoder-only scheduler step、其间推进已有文本 D、随后图文 P 才记为有效；token ID 不一致或任何校验错误均明确保留并单列。",
              "", f"取消样本：客户端断开记录={cancellation['client_cancel_record'] is not None}，图像请求随后由 scheduler finished_req_ids 回收={cancellation['finished_after_client_cancel']}，4 路文本 token 数={cancellation['text_token_counts']}。",
              "", f"详细逐事件证据：`{out_json}`。"]
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "cells": [{
            "label": cell["label"],
            "status": cell["status"],
            "episodes": [{
                "episode_id": episode["episode_id"],
                "errors": episode["validation_errors"],
                "text_tokens": episode["text_token_counts"],
                "image_tokens": episode["image_token_count"],
                "output_ids_equal_to_native": episode["output_ids_equal_to_native"],
                "split_steps": len(episode["split_steps"]),
                "split_text_d": episode["split_steps"][0]["text_decode_tokens"] if episode["split_steps"] else {},
                "first_P": episode["first_P_after_split"],
                "visual_E_between_split_and_P": len(episode["visual_forward_between_split_and_P"]),
                "finished_events": len(episode["image_finished_events"]),
            } for episode in cell["episodes"]],
            "writer": {k: (cell["async_writer"] or {}).get(k) for k in ("submitted", "written", "dropped", "write_errors", "closed_cleanly")},
        } for cell in cells],
        "cancellation": {
            "label": cancellation["label"],
            "partial_image_tokens": cancellation["partial_image_tokens"],
            "split": cancellation["split_encoder_only_step"] is not None,
            "first_P": cancellation["first_P_step"],
            "visual_E_count": len(cancellation["visual_forward_between_split_and_P"]),
            "finished_after_cancel": cancellation["finished_after_client_cancel"],
            "text_token_counts": cancellation["text_token_counts"],
            "writer": {k: (cancellation["async_writer"] or {}).get(k) for k in ("submitted", "written", "dropped", "write_errors", "closed_cleanly")},
            "runner_status": cancellation["runner_status"],
        },
        "files": [str(out_json), str(out_md)],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
