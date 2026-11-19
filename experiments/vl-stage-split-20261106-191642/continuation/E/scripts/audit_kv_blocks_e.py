#!/usr/bin/env python3
"""Reconstruct request-scoped KV block occupancy from E scheduler traces."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSONL {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def block_groups(value: Any) -> list[set[int]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"block ids must be a nested list or null, got {type(value).__name__}")
    return [{int(block_id) for block_id in group} for group in value]


def ids_by_group(value: Any) -> list[set[int]]:
    if value is None:
        return []
    return block_groups(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.scheduler_jsonl)
    errors: list[str] = []
    geometry: dict[int, dict[str, Any]] | None = None
    capacity_blocks: int | None = None
    live: dict[str, dict[str, Any]] = {}
    request_summaries: dict[str, dict[str, Any]] = {}
    occupancy_rows: list[dict[str, Any]] = []
    max_live_blocks_by_group: dict[int, int] = {}
    max_live_bytes = 0

    def ensure_group_count(blocks: list[set[int]], request_id: str) -> list[set[int]]:
        assert geometry is not None
        expected = len(geometry)
        if len(blocks) != expected:
            errors.append(
                f"request {request_id} has {len(blocks)} block groups; KV geometry has {expected}"
            )
            blocks = blocks[:expected] + [set() for _ in range(max(0, expected - len(blocks)))]
        return blocks

    def request_state(request_id: str) -> dict[str, Any]:
        state = live.get(request_id)
        if state is None:
            state = {
                "blocks_by_group": [set() for _ in range(len(geometry or {}))],
                "prompt_len": None,
                "num_computed_tokens": 0,
                "num_output_tokens": 0,
                "last_scheduled_context_tokens": 0,
                "first_row": None,
                "last_row": None,
                "max_reserved_bytes": 0,
                "max_block_counts_by_group": [0 for _ in range(len(geometry or {}))],
                "profile_request": request_id.startswith("_warmup_"),
            }
            live[request_id] = state
        return state

    def record_summary(request_id: str, state: dict[str, Any], row_idx: int) -> None:
        assert geometry is not None
        group_counts = [len(group) for group in state["blocks_by_group"]]
        reserved_bytes_by_group = {
            str(group_idx): count * geometry[group_idx]["bytes_per_logical_block"]
            for group_idx, count in enumerate(group_counts)
        }
        reserved_bytes = sum(reserved_bytes_by_group.values())
        state["max_reserved_bytes"] = max(state["max_reserved_bytes"], reserved_bytes)
        state["max_block_counts_by_group"] = [
            max(previous, current)
            for previous, current in zip(state["max_block_counts_by_group"], group_counts)
        ]
        state["first_row"] = row_idx if state["first_row"] is None else state["first_row"]
        state["last_row"] = row_idx
        summary = request_summaries.setdefault(
            request_id,
            {
                "request_id": request_id,
                "profile_request": state["profile_request"],
                "prompt_len": state["prompt_len"],
                "first_row": row_idx,
                "last_row": row_idx,
                "peak_reserved_bytes": 0,
                "peak_block_counts_by_group": [0 for _ in range(len(geometry))],
                "samples": [],
            },
        )
        summary["last_row"] = row_idx
        summary["peak_reserved_bytes"] = max(summary["peak_reserved_bytes"], reserved_bytes)
        summary["peak_block_counts_by_group"] = [
            max(previous, current)
            for previous, current in zip(summary["peak_block_counts_by_group"], group_counts)
        ]
        summary["samples"].append(
            {
                "row_index": row_idx,
                "time_ns": None,
                "num_computed_tokens": int(state["num_computed_tokens"]),
                "num_output_tokens": int(state["num_output_tokens"]),
                "scheduled_context_tokens": int(state["last_scheduled_context_tokens"]),
                "block_counts_by_group": group_counts,
                "reserved_bytes_by_group": reserved_bytes_by_group,
                "reserved_bytes": reserved_bytes,
                "block_capacity_tokens_by_group": {
                    str(group_idx): count * geometry[group_idx]["block_size_tokens"]
                    for group_idx, count in enumerate(group_counts)
                },
            }
        )

    for row_idx, row in enumerate(rows):
        if row.get("kind") != "scheduler_step":
            continue
        kv = row.get("kv_cache_groups")
        if kv is not None:
            current_capacity = int(kv.get("num_blocks", 0))
            current_geometry = {
                int(group["group_index"]): {
                    "spec_type": group.get("spec_type"),
                    "block_size_tokens": int(group["block_size_tokens"]),
                    "bytes_per_logical_block": int(group["bytes_per_logical_block"]),
                    "layer_count": int(group["layer_count"]),
                    "layer_names": group.get("layer_names", []),
                    "dtype": group.get("dtype"),
                }
                for group in kv.get("groups", [])
                if group.get("block_size_tokens") is not None
                and group.get("bytes_per_logical_block") is not None
            }
            if geometry is None:
                geometry = current_geometry
                capacity_blocks = current_capacity
            elif current_geometry != geometry or current_capacity != capacity_blocks:
                errors.append(f"KV geometry changed at scheduler row {row_idx}")
        if geometry is None:
            errors.append(f"scheduler row {row_idx} lacks usable KV cache geometry")
            continue

        scheduled_tokens = {
            str(request_id): int(num_tokens)
            for request_id, num_tokens in row.get("num_scheduled_tokens", {}).items()
        }
        new_reqs = row.get("scheduled_new_reqs", [])
        cached_reqs = row.get("scheduled_cached_reqs", [])

        for req in new_reqs:
            request_id = str(req["request_id"])
            if request_id in live:
                errors.append(f"request {request_id} allocated as new while already live at row {row_idx}")
            state = request_state(request_id)
            state["blocks_by_group"] = ensure_group_count(
                block_groups(req.get("block_ids_by_group")), request_id
            )
            state["prompt_len"] = int(req.get("prompt_len", 0))
            state["num_computed_tokens"] = int(req.get("num_computed_tokens", 0))
            state["num_output_tokens"] = 0
            state["last_scheduled_context_tokens"] = (
                state["num_computed_tokens"] + scheduled_tokens.get(request_id, 0)
            )

        for req in cached_reqs:
            request_id = str(req["request_id"])
            state = request_state(request_id)
            state["blocks_by_group"] = ensure_group_count(state["blocks_by_group"], request_id)
            new_blocks = req.get("new_block_ids_by_group")
            if bool(req.get("resumed")):
                if new_blocks is None:
                    errors.append(f"resumed request {request_id} lacks replacement block IDs at row {row_idx}")
                else:
                    state["blocks_by_group"] = ensure_group_count(
                        block_groups(new_blocks), request_id
                    )
            elif new_blocks is not None:
                additions = ensure_group_count(block_groups(new_blocks), request_id)
                for group_idx, ids in enumerate(additions):
                    state["blocks_by_group"][group_idx].update(ids)
            state["num_computed_tokens"] = int(req.get("num_computed_tokens", 0))
            state["num_output_tokens"] = int(req.get("num_output_tokens", 0))
            state["last_scheduled_context_tokens"] = (
                state["num_computed_tokens"] + scheduled_tokens.get(request_id, 0)
            )

        # Any request scheduled this step should now have block state and a token count.
        for request_id, num_tokens in scheduled_tokens.items():
            state = live.get(request_id)
            if state is None:
                errors.append(f"scheduled request {request_id} has no allocation state at row {row_idx}")
                continue
            state["last_scheduled_context_tokens"] = (
                int(state["num_computed_tokens"]) + int(num_tokens)
            )
            group_counts = [len(group) for group in state["blocks_by_group"]]
            for group_idx, count in enumerate(group_counts):
                block_size = geometry[group_idx]["block_size_tokens"]
                required_blocks = math.ceil(
                    state["last_scheduled_context_tokens"] / block_size
                )
                if count < required_blocks:
                    errors.append(
                        f"request {request_id} has {count} KV blocks but {required_blocks} are needed "
                        f"for {state['last_scheduled_context_tokens']} scheduled context tokens at row {row_idx}"
                    )
            record_summary(request_id, state, row_idx)

        # Include allocations held by active, unscheduled requests in live occupancy.
        allocated_by_group: dict[int, set[int]] = {idx: set() for idx in geometry}
        for request_id, state in live.items():
            for group_idx, ids in enumerate(state["blocks_by_group"]):
                duplicates = allocated_by_group[group_idx].intersection(ids)
                if duplicates:
                    errors.append(
                        f"KV block IDs are shared by active requests in group {group_idx} at row {row_idx}: "
                        f"{sorted(duplicates)[:8]}"
                    )
                allocated_by_group[group_idx].update(ids)
        group_counts = {str(idx): len(ids) for idx, ids in allocated_by_group.items()}
        group_bytes = {
            str(idx): len(allocated_by_group[idx]) * geometry[idx]["bytes_per_logical_block"]
            for idx in geometry
        }
        total_bytes = sum(group_bytes.values())
        max_live_bytes = max(max_live_bytes, total_bytes)
        for idx, count in allocated_by_group.items():
            max_live_blocks_by_group[idx] = max(max_live_blocks_by_group.get(idx, 0), len(count))
            if capacity_blocks is not None and len(count) > capacity_blocks:
                errors.append(f"KV group {idx} exceeds {capacity_blocks} physical blocks at row {row_idx}")
        occupancy_rows.append(
            {
                "row_index": row_idx,
                "time_ns": row.get("time_ns"),
                "active_requests": sum(not request_id.startswith("_warmup_") for request_id in live),
                "profile_requests": sum(request_id.startswith("_warmup_") for request_id in live),
                "allocated_blocks_by_group": group_counts,
                "reserved_bytes_by_group": group_bytes,
                "reserved_bytes": total_bytes,
                "capacity_blocks": capacity_blocks,
                "finished_req_ids": [str(x) for x in row.get("finished_req_ids", [])],
                "preempted_req_ids": [str(x) for x in row.get("preempted_req_ids", [])],
            }
        )

        for request_id in set(map(str, row.get("finished_req_ids", []))) | set(
            map(str, row.get("preempted_req_ids", []))
        ):
            live.pop(request_id, None)

    if not rows:
        errors.append("scheduler trace has no rows")
    summary = {
        "schema_version": 1,
        "valid": not errors,
        "source_scheduler_jsonl": str(args.scheduler_jsonl),
        "scheduler_rows": len(rows),
        "kv_geometry": {
            "num_blocks_per_group": capacity_blocks,
            "groups": geometry or {},
        },
        "max_live_blocks_by_group": {str(k): v for k, v in max_live_blocks_by_group.items()},
        "peak_reserved_kv_bytes": max_live_bytes,
        "ending_live_request_ids": sorted(live),
        "request_count": len(request_summaries),
        "requests": request_summaries,
        "occupancy_rows": occupancy_rows,
        "errors": errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "valid": summary["valid"],
                "scheduler_rows": len(rows),
                "requests": len(request_summaries),
                "peak_reserved_kv_bytes": max_live_bytes,
                "ending_live_requests": len(live),
                "errors": errors[:12],
                "error_count": len(errors),
            },
            ensure_ascii=False,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
