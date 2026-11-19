#!/usr/bin/env python3
"""Offline A-stage metrics and event alignment from archived raw JSONL."""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
RESULTS = ART / "remote_project" / "results"
CLIENT = ART / "remote_ssd" / "logs"
CELLS = [
    ("eager", 0, "A-eager-probe0.json", "eager-probe0-client-events.jsonl", None, None),
    ("eager", 1, "A-eager-probe1-light.json", "eager-probe1-light-client-events.jsonl",
     "eager-probe1-light-server-events.jsonl", "eager-probe1-light-scheduler-events.jsonl"),
    ("default", 0, "A-default-probe0.json", "default-probe0-client-events.jsonl", None, None),
    ("default", 1, "A-default-probe1-light-ssd-cache.json",
     "default-probe1-light-ssd-cache-client-events.jsonl",
     "default-probe1-light-ssd-cache-server-events.jsonl",
     "default-probe1-light-ssd-cache-scheduler-events.jsonl"),
]

def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]

def percentile(values: list[float], p: float = 0.95) -> float | None:
    if not values:
        return None
    items = sorted(values)
    x = (len(items) - 1) * p
    lo = int(x)
    hi = min(lo + 1, len(items) - 1)
    return items[lo] + (items[hi] - items[lo]) * (x - lo)

def block_index(value: str) -> str:
    found = re.search(r"-(\d+)$", value)
    return found.group(1) if found else value

def window_gap(events: list[dict[str, Any]], anchor: int) -> tuple[float | None, dict[str, Any] | None, int]:
    lo = anchor - 50_000_000
    hi = anchor + 400_000_000
    by_lane: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for event in events:
        if event.get("kind") != "sse_chunk" or event.get("lane") is None:
            continue
        token_ids = event.get("token_ids") or []
        if not token_ids:
            continue
        t = int(event["time_ns"])
        by_lane[int(event["lane"])].append((t, len(token_ids)))
    best_ms = None
    best = None
    token_times = 0
    for lane, chunks in by_lane.items():
        chunks.sort(key=lambda row: row[0])
        times: list[int] = []
        for t, count in chunks:
            times.extend([t] * max(1, count))
        token_times += len(times)
        for before, after in zip(times, times[1:]):
            left, right = max(before, lo), min(after, hi)
            if right <= left:
                continue
            gap = (right - left) / 1e6
            if best_ms is None or gap > best_ms:
                best_ms = gap
                best = {
                    "lane": lane,
                    "start_ns": before,
                    "end_ns": after,
                    "window_start_ns": left,
                    "window_end_ns": right,
                    "gap_ms": gap,
                    "left_censored": before < lo,
                    "right_censored": after > hi,
                }
    return best_ms, best, token_times

def req_matches(key: str, request_id: str | None) -> bool:
    return bool(request_id and key.startswith(request_id + "-"))

def load_cell(config: str, probe: int, result_file: str, client_file: str,
              server_file: str | None, scheduler_file: str | None) -> dict[str, Any]:
    result_data = json.loads((RESULTS / result_file).read_text())
    episodes = [row for row in result_data["episodes"] if not row.get("diagnostic")]
    client_rows = read_jsonl(CLIENT / client_file)
    events_by_ep: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in client_rows:
        ep = event.get("episode_id")
        if ep:
            events_by_ep[ep].append(event)
    server_rows = read_jsonl(CLIENT / server_file if server_file else None)
    scheduler_rows = read_jsonl(CLIENT / scheduler_file if scheduler_file else None)
    anchors = {
        event["episode_id"]: event
        for event in client_rows
        if event.get("kind") == "episode_anchor" and event.get("episode_id")
    }
    gaps: dict[tuple[str, str], list[float]] = defaultdict(list)
    ep_stats = {}
    validation = {"episodes": len(episodes), "invalid": 0, "text_tokens": 0, "image_tokens": 0}
    for ep in episodes:
        eid = ep["episode_id"]
        arm = ep["arm"]
        errors = ep.get("validation_errors") or []
        if errors or ep.get("error") or len(ep.get("text", [])) != 4:
            validation["invalid"] += 1
        validation["text_tokens"] += sum(int(x.get("token_count") or 0) for x in ep.get("text", []))
        if ep.get("image"):
            validation["image_tokens"] += int(ep["image"].get("token_count") or 0)
        anchor = anchors.get(eid, {})
        gap, interval, token_times = window_gap(
            events_by_ep.get(eid, []), int(anchor.get("time_ns", ep.get("anchor_ns", 0)))
        )
        if gap is not None:
            b = block_index(str(ep.get("block")))
            gaps[(b, arm)].append(gap)
        ep_stats[eid] = {
            "block": str(ep.get("block")),
            "block_index": block_index(str(ep.get("block"))),
            "arm": arm,
            "anchor_ns": int(anchor.get("time_ns", ep.get("anchor_ns", 0))),
            "gap_ms": gap,
            "max_gap": interval,
            "tokens_in_window": token_times,
            "image_request_id": (ep.get("image") or {}).get("request_id"),
            "text_request_ids": {
                str(x.get("lane")): x.get("request_id")
                for x in events_by_ep.get(eid, [])
                if x.get("kind") == "sse_chunk" and x.get("lane") is not None and x.get("request_id")
            },
        }
    block_summary = {}
    for b in ("1", "2"):
        block_summary[b] = {}
        for arm in ("T", "R"):
            vals = gaps.get((b, arm), [])
            block_summary[b][arm] = {
                "n": len(vals), "p95_ms": percentile(vals), "median_ms": percentile(vals, 0.5),
                "values_ms": vals,
            }
    return {
        "config": config, "probe": probe, "result_file": result_file, "client_file": client_file,
        "status": result_data.get("status"), "validation": validation, "episodes": ep_stats,
        "block_summary": block_summary,
        "pooled_p95_ms": {
            arm: percentile([v for (b, a), values in gaps.items() if a == arm for v in values])
            for arm in ("T", "R")
        },
        "server_events": server_rows, "scheduler_events": scheduler_rows,
        "event_kind_counts": {
            kind: sum(1 for row in server_rows if row.get("kind") == kind)
            for kind in sorted({row.get("kind", "unknown") for row in server_rows})
        },
    }

def main() -> None:
    cells = [load_cell(*cell) for cell in CELLS]
    cell_by_key = {(x["config"], x["probe"]): x for x in cells}
    overhead = {}
    for config in ("eager", "default"):
        off = cell_by_key[(config, 0)]
        on = cell_by_key[(config, 1)]
        overhead[config] = {}
        for arm in ("T", "R"):
            off_p95, on_p95 = off["pooled_p95_ms"][arm], on["pooled_p95_ms"][arm]
            diff = on_p95 - off_p95 if off_p95 is not None and on_p95 is not None else None
            limit = max(1.0, 0.05 * off_p95) if off_p95 is not None else None
            overhead[config][arm] = {
                "probe_off_p95_ms": off_p95, "probe_on_p95_ms": on_p95,
                "signed_difference_ms": diff,
                "absolute_difference_ms": abs(diff) if diff is not None else None,
                "allowed_difference_ms": limit,
                "pass": abs(diff) <= limit if diff is not None and limit is not None else False,
            }
    gate_by_probe = {}
    for probe in (0, 1):
        cell = cell_by_key[("default", probe)]
        blocks, all_pass = {}, True
        for b in ("1", "2"):
            vals = cell["block_summary"][b]
            t, r = vals["T"]["p95_ms"], vals["R"]["p95_ms"]
            diff = r - t if t is not None and r is not None else None
            relative = diff / t if diff is not None and t else None
            ok = bool(diff is not None and diff >= 3.0 and relative is not None and relative >= 0.15)
            all_pass &= ok
            blocks[b] = {"T_p95_ms": t, "R_p95_ms": r, "R_minus_T_ms": diff,
                         "relative_to_T": relative, "pass": ok}
        gate_by_probe[str(probe)] = {"blocks": blocks, "both_blocks_pass": all_pass}

    on_cell = cell_by_key[("default", 1)]
    client_events = read_jsonl(CLIENT / on_cell["client_file"])
    client_by_ep: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in client_events:
        if event.get("episode_id"):
            client_by_ep[event["episode_id"]].append(event)
    server, scheduler = on_cell["server_events"], on_cell["scheduler_events"]
    alignment = []
    for eid, stat in on_cell["episodes"].items():
        if stat["arm"] != "R":
            continue
        events = client_by_ep.get(eid, [])
        image_id, anchor_ns = stat["image_request_id"], stat["anchor_ns"]
        text_ids = set(stat["text_request_ids"].values())
        image_sse = [x["time_ns"] for x in events if x.get("kind") == "sse_chunk" and x.get("lane") is None and x.get("token_ids")]
        image_first = min(image_sse) if image_sse else None
        first_mix = None
        for row in scheduler:
            t = int(row.get("time_ns", 0))
            if t < anchor_ns or (image_first is not None and t > image_first):
                continue
            ids = list((row.get("num_scheduled_tokens") or {}).keys())
            image_present = any(req_matches(key, image_id) for key in ids)
            text_present = any(req_matches(key, rid) for key in ids for rid in text_ids)
            if image_present and text_present and (first_mix is None or t < int(first_mix["time_ns"])):
                first_mix = row
        image_index = next((x.get("image_index") for x in events if x.get("kind") == "episode_anchor"), None)
        visual = [
            row for row in server
            if row.get("kind") == "visual_forward" and row.get("mode") == "R"
            and row.get("image_index") == image_index
            and anchor_ns - 50_000_000 <= int(row.get("time_ns", 0)) <= (image_first or anchor_ns + 2_000_000_000)
        ]
        visual_row = min(visual, key=lambda x: abs(int(x["time_ns"]) - anchor_ns)) if visual else None
        gap = stat.get("max_gap") or {}
        alignment.append({
            "episode_id": eid, "block": stat["block_index"], "image_request_id": image_id,
            "text_request_ids": sorted(text_ids), "anchor_ns": anchor_ns, "max_gap": gap,
            "first_mixed_step_ns": int(first_mix["time_ns"]) if first_mix else None,
            "first_mixed_num_scheduled_tokens": first_mix.get("num_scheduled_tokens") if first_mix else None,
            "visual_forward_ns": int(visual_row["time_ns"]) if visual_row else None,
            "visual_forward_elapsed_ms": visual_row.get("cpu_elapsed_ms") if visual_row else None,
            "first_image_client_token_ns": image_first,
            "mixed_step_inside_max_gap": bool(first_mix and gap and gap["start_ns"] <= int(first_mix["time_ns"]) <= gap["end_ns"]),
            "visual_forward_inside_max_gap": bool(visual_row and gap and gap["start_ns"] <= int(visual_row["time_ns"]) <= gap["end_ns"]),
        })
    valid_alignment = [x for x in alignment if x["first_mixed_step_ns"] is not None]
    report = {
        "method": {
            "window": "anchor -50ms to anchor +400ms",
            "gap_calculation": "Per episode and lane, maximum overlap of a consecutive client SSE token arrival interval with the fixed window; boundary-crossing intervals are clipped to the window; episode P95 uses linear interpolation.",
            "p95": "linear interpolation at (n-1)*0.95",
            "probe_overhead": "absolute difference of pooled per-episode P95, allowed max(1ms,5% of probe-off P95)",
            "alignment_join": "client image/text response ids matched to worker scheduler request-id prefixes; visual events matched by image_index and time",
        },
        "cells": [
            {k: v for k, v in cell.items() if k not in ("episodes", "server_events", "scheduler_events")}
            for cell in cells
        ],
        "probe_overhead": overhead,
        "default_block_gate": gate_by_probe,
        "default_probe_on_alignment": {
            "R_episodes": len(alignment),
            "first_mixed_step_found": len(valid_alignment),
            "max_gap_contains_first_mixed_step": sum(x["mixed_step_inside_max_gap"] for x in alignment),
            "max_gap_contains_visual_forward": sum(x["visual_forward_inside_max_gap"] for x in alignment),
            "episodes": alignment,
        },
        "excluded_cells": [
            "A-eager-probe1-heavy.json: exploratory over-instrumented run, 4s/episode and API output log volume too high.",
            "A-default-probe1-homecache.json: default cache was first written under /home/chx/.cache; rerun under VLLM_CACHE_ROOT on SSD is the formal cell.",
        ],
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "A-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# A 阶段离线复算", "", "## 分块 P95（ms）", "", "|配置|probe|块|T P95|R P95|R−T|相对T|", "|---|---:|---:|---:|---:|---:|---:|"]
    for probe in (0, 1):
        cell = cell_by_key[("default", probe)]
        for b in ("1", "2"):
            v = cell["block_summary"][b]
            t, r = v["T"]["p95_ms"], v["R"]["p95_ms"]
            d = r - t if t is not None and r is not None else None
            rel = d / t if d is not None and t else None
            lines.append(f"|default|{probe}|{b}|{t:.3f}|{r:.3f}|{d:.3f}|{rel:.1%}|")
    lines += ["", "## 插桩开销", "", "|配置|臂|probe-off P95|probe-on P95|绝对差|允许差|通过|", "|---|---|---:|---:|---:|---:|---|"]
    for config in ("eager", "default"):
        for arm in ("T", "R"):
            row = overhead[config][arm]
            lines.append(f"|{config}|{arm}|{row['probe_off_p95_ms']:.3f}|{row['probe_on_p95_ms']:.3f}|{row['absolute_difference_ms']:.3f}|{row['allowed_difference_ms']:.3f}|{row['pass']}|")
    lines += ["", "## 首混合步事件关联", "",
              f"- 图文 R episode：{len(alignment)}；找到首个混合 scheduler step：{len(valid_alignment)}。",
              f"- 最大文本间隔包含首混合 step：{sum(x['mixed_step_inside_max_gap'] for x in alignment)}。",
              f"- 最大文本间隔包含视觉 forward 起点：{sum(x['visual_forward_inside_max_gap'] for x in alignment)}。",
              "- 详细原始事件关联见 results/A-analysis.json。"]
    (ROOT / "results" / "A-analysis.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "default_gate": gate_by_probe,
        "probe_overhead": overhead,
        "alignment": report["default_probe_on_alignment"] | {"episodes": alignment[:3]},
        "files": [str(ROOT / "results" / "A-analysis.json"), str(ROOT / "results" / "A-analysis.md")]
    }, indent=2))

if __name__ == "__main__":
    main()
