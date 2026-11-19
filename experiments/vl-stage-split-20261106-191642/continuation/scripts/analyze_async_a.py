#!/usr/bin/env python3
"""Summarize the continuation A probe-off/on episodes from immutable raw logs."""
from __future__ import annotations

import json
import os
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "continuation" / "raw"
EXTRACTED = RAW / "async_a"


def ensure_extracted() -> None:
    archive = RAW / "A-async-evidence.tar.gz"
    if not EXTRACTED.exists():
        EXTRACTED.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as tar:
        root = EXTRACTED.resolve()
        for member in tar.getmembers():
            target = (EXTRACTED / member.name).resolve()
            if os.path.commonpath((str(root), str(target))) != str(root):
                raise ValueError(f"archive member escapes destination: {member.name}")
            tar.extract(member, EXTRACTED)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], p: float = 0.95) -> float | None:
    if not values:
        return None
    items = sorted(values)
    index = (len(items) - 1) * p
    lo = int(index)
    hi = min(lo + 1, len(items) - 1)
    return items[lo] + (items[hi] - items[lo]) * (index - lo)


def block_index(value: str) -> str:
    found = re.search(r"-(\d+)$", value)
    return found.group(1) if found else value


def window_gap(events: list[dict[str, Any]], anchor: int) -> dict[str, Any] | None:
    lo, hi = anchor - 50_000_000, anchor + 400_000_000
    by_lane: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for event in events:
        if event.get("kind") != "sse_chunk" or event.get("lane") is None:
            continue
        ids = event.get("token_ids") or []
        if ids:
            by_lane[int(event["lane"])].append((int(event["time_ns"]), len(ids)))
    best = None
    for lane, chunks in by_lane.items():
        chunks.sort(key=lambda row: row[0])
        token_times = [t for t, n in chunks for _ in range(max(1, n))]
        for before, after in zip(token_times, token_times[1:]):
            left, right = max(before, lo), min(after, hi)
            if right <= left:
                continue
            gap = (right - left) / 1e6
            if best is None or gap > best["gap_ms"]:
                best = {
                    "lane": lane,
                    "start_ns": before,
                    "end_ns": after,
                    "gap_ms": gap,
                    "left_censored": before < lo,
                    "right_censored": after > hi,
                }
    return best


def load_probe(probe: int) -> dict[str, Any]:
    label = f"A-resume-async-probe{probe}"
    result = json.loads((EXTRACTED / "results" / f"{label}.json").read_text())
    client = read_jsonl(EXTRACTED / "logs" / f"{label}-client-events.jsonl")
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    anchors: dict[str, dict[str, Any]] = {}
    for row in client:
        eid = row.get("episode_id")
        if eid:
            by_episode[eid].append(row)
        if row.get("kind") == "episode_anchor" and eid:
            anchors[eid] = row

    episodes = [row for row in result.get("episodes", []) if not row.get("diagnostic")]
    gaps: dict[tuple[str, str], list[float]] = defaultdict(list)
    summaries: dict[str, dict[str, Any]] = {}
    invalid = 0
    coverage = {"expected_text_tokens": 0, "client_text_tokens": 0,
                "expected_image_tokens": 0, "client_image_tokens": 0}
    for episode in episodes:
        eid = episode["episode_id"]
        observed = by_episode.get(eid, [])
        anchor = int(anchors.get(eid, {}).get("time_ns", episode.get("anchor_ns", 0)))
        gap = window_gap(observed, anchor)
        block = block_index(str(episode.get("block", "unknown")))
        arm = str(episode.get("arm", "unknown"))
        if gap:
            gaps[(block, arm)].append(float(gap["gap_ms"]))
        expected_text = sum(int(row.get("token_count") or 0) for row in episode.get("text", []))
        expected_image = int((episode.get("image") or {}).get("token_count") or 0)
        client_text = sum(len(row.get("token_ids") or []) for row in observed
                          if row.get("kind") == "sse_chunk" and row.get("lane") is not None)
        client_image = sum(len(row.get("token_ids") or []) for row in observed
                           if row.get("kind") == "sse_chunk" and row.get("lane") is None)
        coverage["expected_text_tokens"] += expected_text
        coverage["client_text_tokens"] += client_text
        coverage["expected_image_tokens"] += expected_image
        coverage["client_image_tokens"] += client_image
        errors = bool(episode.get("validation_errors") or episode.get("error"))
        errors |= len(episode.get("text", [])) != 4 or expected_text != 4 * 192
        errors |= expected_image != (16 if arm == "R" else 0)
        errors |= expected_text != client_text or expected_image != client_image
        invalid += int(errors)
        image = episode.get("image") or {}
        summaries[eid] = {
            "block": block,
            "arm": arm,
            "gap_ms": gap["gap_ms"] if gap else None,
            "max_gap": gap,
            "image_request_id": image.get("request_id"),
            "image_sha256": image.get("image_sha256"),
            "invalid": errors,
        }

    blocks: dict[str, dict[str, Any]] = {}
    for block in sorted({b for b, _ in gaps}, key=lambda x: int(x) if x.isdigit() else x):
        blocks[block] = {}
        for arm in ("T", "R"):
            values = gaps.get((block, arm), [])
            blocks[block][arm] = {
                "n": len(values),
                "p95_ms": percentile(values),
                "median_ms": percentile(values, 0.5),
            }
    pooled = {
        arm: percentile([v for (b, a), values in gaps.items() if a == arm for v in values])
        for arm in ("T", "R")
    }

    log_dir = EXTRACTED / "logs"
    server_path = log_dir / f"{label}-server-events.jsonl"
    scheduler_path = log_dir / f"{label}-scheduler-events.jsonl"
    server = read_jsonl(server_path) if server_path.exists() else []
    scheduler = read_jsonl(scheduler_path) if scheduler_path.exists() else []
    meta_path = log_dir / f"{label}-server-events.jsonl.async-meta.json"
    async_meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    server_kinds = Counter(row.get("kind", "unknown") for row in server)
    scheduler_kinds = Counter(row.get("kind", "unknown") for row in scheduler)
    return {
        "probe": probe,
        "status": result.get("status"),
        "measurement_episodes": len(episodes),
        "diagnostic_episodes": sum(bool(x.get("diagnostic")) for x in result.get("episodes", [])),
        "arm_counts": dict(Counter(x.get("arm") for x in episodes)),
        "invalid_episodes": invalid,
        "token_coverage": coverage,
        "blocks": blocks,
        "pooled_p95_ms": pooled,
        "episode_summary": summaries,
        "async_writer": async_meta,
        "server_event_lines": len(server),
        "server_event_kinds": dict(server_kinds.most_common()),
        "scheduler_event_lines": len(scheduler),
        "scheduler_event_kinds": dict(scheduler_kinds.most_common()),
    }


def main() -> None:
    ensure_extracted()
    off = load_probe(0)
    on = load_probe(1)
    overhead = {}
    effect = {}
    for arm in ("T", "R"):
        a, b = off["pooled_p95_ms"][arm], on["pooled_p95_ms"][arm]
        delta = b - a if a is not None and b is not None else None
        overhead[arm] = {
            "probe_off_p95_ms": a,
            "probe_on_p95_ms": b,
            "signed_delta_ms": delta,
            "absolute_delta_ms": abs(delta) if delta is not None else None,
            "allowed_delta_ms": max(1.0, 0.05 * a) if a is not None else None,
        }
    for name, cell in (("probe_off", off), ("probe_on", on)):
        effect[name] = {}
        for block, arms in cell["blocks"].items():
            t, r = arms["T"]["p95_ms"], arms["R"]["p95_ms"]
            effect[name][block] = {"T_p95_ms": t, "R_p95_ms": r,
                                   "R_minus_T_ms": r - t if t is not None and r is not None else None}
    report = {
        "method": {
            "window": "anchor -50ms to anchor +400ms",
            "p95": "linear interpolation at (n-1)*0.95 over per-episode maximum text-token gap",
            "diagnostic_episode_excluded": True,
            "scope": "one continuation probe-off/on run; descriptive, not a new independent replication of the original 256-episode set",
        },
        "probe_off": off,
        "probe_on": on,
        "probe_overhead": overhead,
        "block_effect": effect,
    }
    out_json = ROOT / "results" / "A-async-analysis.json"
    out_md = ROOT / "results" / "A-async-analysis.md"
    out_json.parent.mkdir(exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# A 阶段异步探针续做复算", "",
             "## 有效 episode 与客户端 token 覆盖", "",
             "|探针|episode|T|R|无效|期望/观测文本 token|期望/观测图文 token|", "|---:|---:|---:|---:|---:|---:|---:|"]
    for cell in (off, on):
        cov = cell["token_coverage"]
        lines.append(f"|{cell['probe']}|{cell['measurement_episodes']}|{cell['arm_counts'].get('T', 0)}|{cell['arm_counts'].get('R', 0)}|{cell['invalid_episodes']}|{cov['expected_text_tokens']}/{cov['client_text_tokens']}|{cov['expected_image_tokens']}/{cov['client_image_tokens']}|")
    lines += ["", "## 分块窗口最大文本间隔 P95（ms）", "", "|探针|块|T P95|R P95|R−T|", "|---:|---:|---:|---:|---:|"]
    for name, cell in (("probe_off", off), ("probe_on", on)):
        for block, arms in cell["blocks"].items():
            t, r = arms["T"]["p95_ms"], arms["R"]["p95_ms"]
            delta = r - t if t is not None and r is not None else None
            lines.append(f"|{cell['probe']}|{block}|{t:.3f}|{r:.3f}|{delta:.3f}|")
    lines += ["", "## Probe-on/off 描述性差异", "", "|臂|off P95|on P95|差值|允许差|", "|---|---:|---:|---:|---:|"]
    for arm, row in overhead.items():
        lines.append(f"|{arm}|{row['probe_off_p95_ms']:.3f}|{row['probe_on_p95_ms']:.3f}|{row['absolute_delta_ms']:.3f}|{row['allowed_delta_ms']:.3f}|")
    meta = on["async_writer"] or {}
    lines += ["", "## 异步队列与事件覆盖", "",
              f"- writer：submitted={meta.get('submitted')}，written={meta.get('written')}，dropped={meta.get('dropped')}，write_errors={meta.get('write_errors')}，closed_cleanly={meta.get('closed_cleanly')}。",
              f"- probe-on server event：{on['server_event_lines']} 行；scheduler event：{on['scheduler_event_lines']} 行。",
              "- 诊断 episode 单独保留在原始 JSON 中，不计入正式窗口统计。",
              "- 这是单轮续做的描述性对照；不得与上轮正式 256 episode 简单混并当作独立重复。",
              "", f"详细逐 episode 数据：`{out_json}`。"]
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"probe_off_p95_ms": off["pooled_p95_ms"],
                      "probe_on_p95_ms": on["pooled_p95_ms"],
                      "probe_overhead": overhead,
                      "blocks": effect,
                      "validity": {"off": off["invalid_episodes"], "on": on["invalid_episodes"]},
                      "async_writer": on["async_writer"],
                      "files": [str(out_json), str(out_md)]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
