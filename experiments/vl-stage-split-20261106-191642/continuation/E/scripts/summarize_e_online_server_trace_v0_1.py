#!/usr/bin/env python3
"""Extract compact per-image visual-forward token counts from a server event log."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-events", type=Path, required=True)
    parser.add_argument("--client-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    client_by_request: dict[str, dict[str, Any]] = {}
    client_by_label: dict[str, dict[str, Any]] = {}
    image_sends: list[dict[str, Any]] = []
    for line in args.client_events.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("kind") == "image_request_send":
            image_sends.append(row)
        if row.get("kind") == "image_response" and row.get("request_id"):
            client_by_request[str(row["request_id"])] = row
            client_by_label[str(row["label"])] = row

    forwards: list[dict[str, Any]] = []
    phase_counts: collections.Counter[str] = collections.Counter()
    token_counts: collections.Counter[int] = collections.Counter()
    startup_forwards = 0
    unjoined_forwards = 0
    used_labels: set[str] = set()
    with args.server_events.open(encoding="utf-8") as stream:
        for line in stream:
            if not line:
                continue
            row = json.loads(line)
            if row.get("kind") != "visual_forward":
                continue
            request_ids = [str(value) for value in (row.get("request_ids") or [])]
            shape = row.get("output_shape")
            retained = int(shape[0]) if isinstance(shape, list) and shape else None
            if request_ids:
                candidates = [(engine_request_id, next((value for request_id, value in client_by_request.items() if engine_request_id == request_id or engine_request_id.startswith(request_id + "-")), None)) for engine_request_id in request_ids]
            elif retained is not None and retained < 4096 and row.get("time_ns") is not None:
                event_time = int(row["time_ns"])
                nearest = min(
                    (send for send in image_sends if str(send.get("request_label")) not in used_labels),
                    key=lambda send: abs(int(send.get("actual_send_ns", event_time)) - event_time),
                    default=None,
                )
                if nearest is not None and abs(int(nearest.get("actual_send_ns", event_time)) - event_time) <= 5_000_000_000:
                    label = str(nearest.get("request_label"))
                    used_labels.add(label)
                    candidates = [("", client_by_label.get(label))]
                else:
                    candidates = []
                    unjoined_forwards += 1
            else:
                if retained is None or retained >= 4096:
                    startup_forwards += 1
                else:
                    unjoined_forwards += 1
                continue

            for engine_request_id, client in candidates:
                if client is None:
                    unjoined_forwards += 1
                    continue
                phase = str(client.get("phase")) if client else "unjoined"
                phase_counts[phase] += 1
                if retained is not None:
                    token_counts[retained] += 1
                forwards.append(
                    {
                        "engine_request_id": engine_request_id,
                        "request_id": client.get("request_id") if client else None,
                        "request_label": client.get("label") if client else None,
                        "phase": phase,
                        "image_id": client.get("image_id") if client else None,
                        "image_sha256": client.get("image_sha256") if client else None,
                        "retained_visual_tokens": retained,
                        "output_shape": shape,
                        "feature_key": row.get("feature_key"),
                        "mm_hash": row.get("mm_hash"),
                        "native_cache_hit": row.get("native_cache_hit"),
                        "time_ns": row.get("time_ns"),
                    }
                )

    output = {
        "schema_version": 1,
        "server_events_path": str(args.server_events),
        "server_events_sha256": sha256(args.server_events),
        "client_events_path": str(args.client_events),
        "client_events_sha256": sha256(args.client_events),
        "visual_forward_count_joined": len(forwards),
        "visual_forward_count_unjoined": unjoined_forwards,
        "startup_visual_forward_count_unjoined": startup_forwards,
        "joined_phase_counts": dict(phase_counts),
        "retained_visual_token_counts": {str(key): value for key, value in sorted(token_counts.items())},
        "visual_forwards": forwards,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "joined_forwards": len(forwards),
                "unjoined_forwards": unjoined_forwards,
                "phase_counts": dict(phase_counts),
                "retained_visual_token_counts": dict(token_counts),
                "startup_unjoined": startup_forwards,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
