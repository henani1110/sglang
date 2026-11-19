#!/usr/bin/env python3
"""Recompute the buffered A-stage pair and align engine/API/client token events."""
from __future__ import annotations

import json
import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], p: float = 0.95) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * p
    low = int(pos)
    high = min(low + 1, len(xs) - 1)
    return xs[low] + (xs[high] - xs[low]) * (pos - low)


def block_index(value: str) -> str:
    match = re.search(r"-(\d+)$", value)
    return match.group(1) if match else value


def window_gap(events: list[dict[str, Any]], anchor: int) -> tuple[float | None, dict[str, Any] | None]:
    lo, hi = anchor - 50_000_000, anchor + 400_000_000
    lanes: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for event in events:
        if event.get("kind") != "sse_chunk" or event.get("lane") is None:
            continue
        token_ids = event.get("token_ids") or []
        if token_ids:
            lanes[int(event["lane"])].append((int(event["time_ns"]), len(token_ids)))
    best_ms, best = None, None
    for lane, chunks in lanes.items():
        chunks.sort(key=lambda item: item[0])
        times = [t for t, count in chunks for _ in range(max(1, count))]
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
    return best_ms, best


def request_matches(key: str, request_id: str | None) -> bool:
    return bool(request_id and key.startswith(request_id + "-"))


def event_for_token(rows: list[dict[str, Any]], request_id: str, token_index: int) -> dict[str, Any] | None:
    matches = [
        row for row in rows
        if row.get("kind") == "engine_result_ready_to_api"
        and row.get("request_id") == request_id
        and row.get("token_index_start") is not None
        and int(row["token_index_start"]) <= token_index <= int(row["token_index_end"])
    ]
    return min(matches, key=lambda row: int(row["time_ns"])) if matches else None


def load_cell(probe: int, variant: str) -> dict[str, Any]:
    raw = ROOT / "artifacts" / f"vl-stage-split-A-{variant}-archive" / "ssd"
    label = f"default-probe{probe}-{variant}"
    result_data = json.loads((raw / "results" / f"{label}.json").read_text())
    client = read_jsonl(raw / "logs" / f"{label}-client-events.jsonl")
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    anchors: dict[str, dict[str, Any]] = {}
    for row in client:
        eid = row.get("episode_id")
        if eid:
            by_episode[eid].append(row)
        if row.get("kind") == "episode_anchor" and eid:
            anchors[eid] = row

    server = read_jsonl(raw / "logs" / f"{label}-server-events.jsonl") if probe else []
    scheduler = read_jsonl(raw / "logs" / f"{label}-scheduler-events.jsonl") if probe else []
    episodes = [row for row in result_data["episodes"] if not row.get("diagnostic")]
    gaps: dict[tuple[str, str], list[float]] = defaultdict(list)
    stats: dict[str, dict[str, Any]] = {}
    invalid = 0
    coverage = {"text_token_ids": 0, "image_token_ids": 0, "expected_text_token_ids": 0, "expected_image_token_ids": 0}

    for ep in episodes:
        eid = ep["episode_id"]
        observed = by_episode[eid]
        anchor = anchors.get(eid, {})
        anchor_ns = int(anchor.get("time_ns", ep.get("anchor_ns", 0)))
        gap, interval = window_gap(observed, anchor_ns)
        b = block_index(str(ep.get("block")))
        if gap is not None:
            gaps[(b, ep["arm"])].append(gap)
        text_ids = [int(t.get("token_count") or 0) for t in ep.get("text", [])]
        image_count = int((ep.get("image") or {}).get("token_count") or 0)
        client_text_counts: dict[str, int] = defaultdict(int)
        client_image_count = 0
        for event in observed:
            if event.get("kind") != "sse_chunk":
                continue
            count = len(event.get("token_ids") or [])
            if event.get("lane") is None:
                client_image_count += count
            else:
                client_text_counts[str(event["lane"])] += count
        expected_text = sum(text_ids)
        actual_text = sum(client_text_counts.values())
        coverage["text_token_ids"] += actual_text
        coverage["image_token_ids"] += client_image_count
        coverage["expected_text_token_ids"] += expected_text
        coverage["expected_image_token_ids"] += image_count
        if (
            ep.get("validation_errors")
            or ep.get("error")
            or len(text_ids) != 4
            or text_ids != [192] * 4
            or (ep["arm"] == "R" and image_count != 16)
            or (ep["arm"] == "T" and image_count != 0)
            or actual_text != expected_text
            or client_image_count != image_count
        ):
            invalid += 1
        req_by_lane = {
            str(event["lane"]): event.get("request_id")
            for event in observed
            if event.get("kind") == "sse_chunk" and event.get("lane") is not None and event.get("request_id")
        }
        stats[eid] = {
            "block": b,
            "arm": ep["arm"],
            "anchor_ns": anchor_ns,
            "gap_ms": gap,
            "max_gap": interval,
            "image_request_id": (ep.get("image") or {}).get("request_id"),
            "text_request_ids": req_by_lane,
            "image_index": anchor.get("image_index"),
            "sse_text_tokens_by_lane": dict(client_text_counts),
            "sse_image_tokens": client_image_count,
        }

    blocks: dict[str, dict[str, Any]] = {}
    for block in ("1", "2"):
        blocks[block] = {}
        for arm in ("T", "R"):
            vals = gaps[(block, arm)]
            blocks[block][arm] = {"n": len(vals), "p95_ms": percentile(vals), "median_ms": percentile(vals, 0.5), "values_ms": vals}
    return {
        "probe": probe,
        "label": label,
        "status": result_data.get("status"),
        "measurement_episodes": len(episodes),
        "validation_invalid_episodes": invalid,
        "client_token_coverage": coverage,
        "episodes": stats,
        "blocks": blocks,
        "pooled_p95_ms": {
            arm: percentile([v for (b, a), values in gaps.items() if a == arm for v in values])
            for arm in ("T", "R")
        },
        "server_events": server,
        "scheduler_events": scheduler,
        "client_events": client,
    }


def align_episode(epid: str, stat: dict[str, Any], client: list[dict[str, Any]], server: list[dict[str, Any]], scheduler: list[dict[str, Any]]) -> dict[str, Any]:
    events = [row for row in client if row.get("episode_id") == epid]
    anchor_ns = stat["anchor_ns"]
    image_id = stat["image_request_id"]
    text_ids = set(stat["text_request_ids"].values())
    img_tokens = [int(row["time_ns"]) for row in events if row.get("kind") == "sse_chunk" and row.get("lane") is None and row.get("token_ids")]
    image_first_ns = min(img_tokens) if img_tokens else None
    mixed = []
    for row in scheduler:
        t = int(row.get("time_ns", 0))
        if t < anchor_ns - 50_000_000 or (image_first_ns is not None and t > image_first_ns):
            continue
        keys = list((row.get("num_scheduled_tokens") or {}).keys())
        has_image = any(request_matches(key, image_id) for key in keys)
        has_text = any(request_matches(key, rid) for key in keys for rid in text_ids)
        if has_image and has_text:
            mixed.append(row)
    first_mix = min(mixed, key=lambda row: int(row["time_ns"])) if mixed else None
    visual = [
        row for row in server
        if row.get("kind") == "visual_forward" and row.get("mode") == "R"
        and row.get("image_index") == stat["image_index"]
        and anchor_ns - 50_000_000 <= int(row.get("time_ns", 0)) <= (image_first_ns or anchor_ns + 2_000_000_000)
    ]
    visual_row = min(visual, key=lambda row: abs(int(row["time_ns"]) - anchor_ns)) if visual else None
    gap = stat["max_gap"]
    if not gap:
        return {"episode_id": epid, "first_mixed_step_found": bool(first_mix), "visual_forward_found": bool(visual_row)}

    lane = str(gap["lane"])
    request_id = stat["text_request_ids"].get(lane)
    lane_events = sorted(
        [row for row in events if row.get("kind") == "sse_chunk" and str(row.get("lane")) == lane and row.get("token_ids")],
        key=lambda row: (int(row["time_ns"]), int(row.get("chunk_index", 0))),
    )
    before_candidates = [row for row in lane_events if int(row["time_ns"]) == int(gap["start_ns"])]
    after_candidates = [row for row in lane_events if int(row["time_ns"]) == int(gap["end_ns"])]

    def token_position(target: dict[str, Any], targets: list[dict[str, Any]]) -> tuple[int, int] | None:
        if not targets:
            return None
        target = targets[-1]
        offset = 0
        for row in lane_events:
            if row is target:
                return offset, offset + len(row.get("token_ids") or []) - 1
            offset += len(row.get("token_ids") or [])
        return None

    left_pos = token_position(before_candidates[-1], before_candidates) if before_candidates else None
    right_pos = token_position(after_candidates[0], after_candidates) if after_candidates else None
    engine_file = server
    api_file = server
    left_engine = event_for_token(engine_file, str(request_id), left_pos[1]) if request_id and left_pos else None
    right_engine = event_for_token(engine_file, str(request_id), right_pos[0]) if request_id and right_pos else None
    left_api = next((row for row in api_file if row.get("kind") == "api_yield" and request_id and row.get("request_id") == request_id and left_pos and int(row.get("token_index_start", -1)) <= left_pos[1] <= int(row.get("token_index_end", -1))), None)
    right_api = next((row for row in api_file if row.get("kind") == "api_yield" and request_id and row.get("request_id") == request_id and right_pos and int(row.get("token_index_start", -1)) <= right_pos[0] <= int(row.get("token_index_end", -1))), None)
    left_client = before_candidates[-1] if before_candidates else None
    right_client = after_candidates[0] if after_candidates else None

    def rel(row: dict[str, Any] | None) -> float | None:
        return (int(row["time_ns"]) - anchor_ns) / 1e6 if row else None

    def gap_ms(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float | None:
        return (int(b["time_ns"]) - int(a["time_ns"])) / 1e6 if a and b else None

    matches = {
        "left_token_id_engine_matches_client": bool(left_engine and left_client and left_engine.get("token_ids") == left_client.get("token_ids")),
        "right_token_id_engine_matches_client": bool(right_engine and right_client and right_engine.get("token_ids") == right_client.get("token_ids")),
        "left_token_id_api_matches_client": bool(left_api and left_client and left_api.get("token_ids") == left_client.get("token_ids")),
        "right_token_id_api_matches_client": bool(right_api and right_client and right_api.get("token_ids") == right_client.get("token_ids")),
    }
    return {
        "episode_id": epid,
        "block": stat["block"],
        "max_gap_lane": int(lane),
        "text_request_id": request_id,
        "anchor_ns": anchor_ns,
        "image_request_id": image_id,
        "image_first_client_token_ms": rel({"time_ns": image_first_ns}) if image_first_ns else None,
        "first_mixed_step_found": bool(first_mix),
        "first_mixed_step_ms": rel(first_mix),
        "first_mixed_scheduled_tokens": first_mix.get("num_scheduled_tokens") if first_mix else None,
        "visual_forward_found": bool(visual_row),
        "visual_forward_ms": rel(visual_row),
        "visual_forward_cpu_elapsed_ms": visual_row.get("cpu_elapsed_ms") if visual_row else None,
        "max_client_gap_ms": gap["gap_ms"],
        "client_gap_start_ms": rel(left_client),
        "client_gap_end_ms": rel(right_client),
        "client_gap_token_indices": [left_pos, right_pos],
        "engine_ready_gap_ms": gap_ms(left_engine, right_engine),
        "engine_ready_start_ms": rel(left_engine),
        "engine_ready_end_ms": rel(right_engine),
        "api_yield_gap_ms": gap_ms(left_api, right_api),
        "api_yield_start_ms": rel(left_api),
        "api_yield_end_ms": rel(right_api),
        "event_join_checks": matches,
        "all_event_joins_match_token_ids": all(matches.values()),
        "mixed_step_precedes_max_gap": bool(first_mix and int(first_mix["time_ns"]) <= int(gap["start_ns"])),
        "visual_forward_precedes_max_gap": bool(visual_row and int(visual_row["time_ns"]) <= int(gap["start_ns"])),
    }


def compare_text_sequences(cell: dict[str, Any]) -> dict[str, Any]:
    formal_ids = set(cell["episodes"])
    expected: dict[str, list[tuple[int, list[int]]]] = defaultdict(list)
    for row in cell["client_events"]:
        if (
            row.get("episode_id") in formal_ids
            and row.get("kind") == "sse_chunk"
            and row.get("lane") is not None
            and row.get("request_id")
            and row.get("token_ids")
        ):
            expected[str(row["request_id"])].append((int(row.get("chunk_index", 0)), [int(x) for x in row["token_ids"]]))
    by_kind: dict[str, dict[str, list[dict[str, Any]]]] = {
        kind: defaultdict(list) for kind in ("engine_result_ready_to_api", "api_yield")
    }
    for row in cell["server_events"]:
        kind = row.get("kind")
        request_id = str(row.get("request_id", ""))
        if kind in by_kind and request_id in expected and row.get("token_ids"):
            by_kind[kind][request_id].append(row)
    summary = {}
    for kind, groups in by_kind.items():
        mismatches = []
        for request_id, parts in expected.items():
            client_ids = [token for _, ids in sorted(parts) for token in ids]
            rows = sorted(groups.get(request_id, []), key=lambda row: int(row["token_index_start"]))
            server_ids = [token for row in rows for token in row["token_ids"]]
            next_index = 0
            contiguous = True
            for row in rows:
                start = int(row["token_index_start"])
                end = int(row["token_index_end"])
                if start != next_index or end - start + 1 != len(row["token_ids"]):
                    contiguous = False
                next_index = end + 1
            if client_ids != server_ids or not contiguous:
                mismatches.append({"request_id": request_id, "client_tokens": len(client_ids), "event_tokens": len(server_ids), "contiguous": contiguous})
        summary[kind] = {
            "formal_text_requests": len(expected),
            "exact_full_token_id_sequence_matches": len(expected) - len(mismatches),
            "mismatches": mismatches,
        }
    summary["scope_note"] = "Engine/API hooks cover the text /v1/completions path. Image chat outputs are retained and counted in client SSE, with image TTFT measured there; no engine/API token hook was installed on the chat endpoint."
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="buffered")
    variant = parser.parse_args().variant
    cells = {probe: load_cell(probe, variant) for probe in (0, 1)}
    off, on = cells[0], cells[1]
    overhead = {}
    for arm in ("T", "R"):
        a, b = off["pooled_p95_ms"][arm], on["pooled_p95_ms"][arm]
        difference = abs(b - a) if a is not None and b is not None else None
        limit = max(1.0, 0.05 * a) if a is not None else None
        overhead[arm] = {
            "probe_off_pooled_p95_ms": a,
            "probe_on_pooled_p95_ms": b,
            "absolute_difference_ms": difference,
            "allowed_difference_ms": limit,
            "pass": bool(difference is not None and limit is not None and difference <= limit),
        }

    gate = {}
    for probe, cell in cells.items():
        blocks = {}
        for block in ("1", "2"):
            t = cell["blocks"][block]["T"]["p95_ms"]
            r = cell["blocks"][block]["R"]["p95_ms"]
            difference = r - t if t is not None and r is not None else None
            relative = difference / t if difference is not None and t else None
            passed = bool(difference is not None and difference >= 3 and relative is not None and relative >= 0.15)
            blocks[block] = {"T_p95_ms": t, "R_p95_ms": r, "R_minus_T_ms": difference, "relative_to_T": relative, "pass": passed}
        gate[str(probe)] = {"blocks": blocks, "both_blocks_pass": all(row["pass"] for row in blocks.values())}

    on_r = [(eid, st) for eid, st in on["episodes"].items() if st["arm"] == "R"]
    aligned = [align_episode(eid, st, on["client_events"], on["server_events"], on["scheduler_events"]) for eid, st in on_r]
    full_sequences = compare_text_sequences(on)
    complete = [row for row in aligned if row.get("engine_ready_gap_ms") is not None and row.get("api_yield_gap_ms") is not None]
    chosen = max((row for row in aligned if row.get("max_client_gap_ms") is not None), key=lambda row: float(row["max_client_gap_ms"]))

    report = {
        "measurement": {
            "probe_off": {k: v for k, v in off.items() if k not in ("episodes", "server_events", "scheduler_events", "client_events")},
            "probe_on": {k: v for k, v in on.items() if k not in ("episodes", "server_events", "scheduler_events", "client_events")},
            "block_gate": gate,
            "probe_overhead": overhead,
        },
        "alignment": {
            "time_domain": "Host monotonic clocks (perf_counter_ns/time.monotonic_ns), same remote host.",
            "R_episodes": len(on_r),
            "first_mixed_scheduler_step_found": sum(bool(row.get("first_mixed_step_found")) for row in aligned),
            "visual_forward_found": sum(bool(row.get("visual_forward_found")) for row in aligned),
            "max_gap_engine_and_api_both_joined": len(complete),
            "token_id_matches_all_event_joins": sum(bool(row.get("all_event_joins_match_token_ids")) for row in complete),
            "full_text_token_sequence_checks": full_sequences,
            "mixed_step_precedes_max_gap": sum(bool(row.get("mixed_step_precedes_max_gap")) for row in aligned),
            "visual_forward_precedes_max_gap": sum(bool(row.get("visual_forward_precedes_max_gap")) for row in aligned),
            "episodes": aligned,
            "illustrative_max_gap_episode": chosen,
        },
        "interpretation_limits": [
            "R−T is a controlled trace-level interference contrast, not a production SLA or an open-loop throughput result.",
            "The max-gap interval is the two client token arrivals defining each episode's maximum in-window text gap; the corresponding engine/API tokens are joined by request ID and token index.",
            "The scheduler mixed step and visual-forward start usually precede the later max-gap interval; their ordering supports a trace association, not single-step causation.",
        ],
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"A-{variant}-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / f"A-{variant}-aligned-timeline.json").write_text(json.dumps(report["alignment"]["illustrative_max_gap_episode"], indent=2) + "\n")

    lines = [
        f"# A 阶段：缓冲式逐 token 复测（{variant}）",
        "",
        "## 分块客户端 P95（ms）",
        "",
        "|probe|块|T n|T P95|R n|R P95|R−T|相对 T|门槛|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for probe, cell in cells.items():
        for block in ("1", "2"):
            row = gate[str(probe)]["blocks"][block]
            lines.append(
                f"|{probe}|{block}|{cell['blocks'][block]['T']['n']}|{row['T_p95_ms']:.3f}|"
                f"{cell['blocks'][block]['R']['n']}|{row['R_p95_ms']:.3f}|{row['R_minus_T_ms']:.3f}|"
                f"{row['relative_to_T']:.1%}|{row['pass']}|"
            )
    lines += ["", "## 插桩开销", "", "|臂|probe-off pooled P95|probe-on pooled P95|绝对差|允许差|通过|", "|---|---:|---:|---:|---:|---|"]
    for arm, row in overhead.items():
        lines.append(
            f"|{arm}|{row['probe_off_pooled_p95_ms']:.3f}|{row['probe_on_pooled_p95_ms']:.3f}|"
            f"{row['absolute_difference_ms']:.3f}|{row['allowed_difference_ms']:.3f}|{row['pass']}|"
        )
    lines += [
        "",
        "## 校验与事件对齐",
        "",
        f"- probe-off：{off['measurement_episodes']} 个测量 episode，校验错误 {off['validation_invalid_episodes']}；SSE 文本 token 覆盖 {off['client_token_coverage']['text_token_ids']}/{off['client_token_coverage']['expected_text_token_ids']}，图像 token 覆盖 {off['client_token_coverage']['image_token_ids']}/{off['client_token_coverage']['expected_image_token_ids']}。",
        f"- probe-on：{on['measurement_episodes']} 个测量 episode，校验错误 {on['validation_invalid_episodes']}；SSE 文本 token 覆盖 {on['client_token_coverage']['text_token_ids']}/{on['client_token_coverage']['expected_text_token_ids']}，图像 token 覆盖 {on['client_token_coverage']['image_token_ids']}/{on['client_token_coverage']['expected_image_token_ids']}。",
        f"- probe-on 的 R episode：{len(on_r)}；首个图文混合 scheduler step {report['alignment']['first_mixed_scheduler_step_found']}；视觉 forward {report['alignment']['visual_forward_found']}；最大间隔的 engine/API 双端 token join {len(complete)}，token ID 一致 {report['alignment']['token_id_matches_all_event_joins']}。",
        f"- 对 128 个正式文本 completion request 逐 token ID 复核：engine-ready {full_sequences['engine_result_ready_to_api']['exact_full_token_id_sequence_matches']}/128 完全一致，API-yield {full_sequences['api_yield']['exact_full_token_id_sequence_matches']}/128 完全一致。图文 chat 输出 token 与 TTFT 由客户端 SSE 计数和计时；hook 未覆盖 chat engine/API endpoint。",
        "",
        "## 代表性对齐时间线",
        "",
        f"选择最大客户端间隔 episode `{chosen['episode_id']}`。时间均相对 episode anchor，单位 ms。",
        "",
        "|事件|相对 anchor|",
        "|---|---:|",
        f"|首个混合 scheduler step|{chosen.get('first_mixed_step_ms')}|",
        f"|visual forward 起点|{chosen.get('visual_forward_ms')}|",
        f"|最大间隔左侧 engine-ready|{chosen.get('engine_ready_start_ms')}|",
        f"|最大间隔左侧 API yield|{chosen.get('api_yield_start_ms')}|",
        f"|最大间隔左侧客户端 SSE|{chosen.get('client_gap_start_ms')}|",
        f"|最大间隔右侧 engine-ready|{chosen.get('engine_ready_end_ms')}|",
        f"|最大间隔右侧 API yield|{chosen.get('api_yield_end_ms')}|",
        f"|最大间隔右侧客户端 SSE|{chosen.get('client_gap_end_ms')}|",
        f"|客户端间隔 / engine-ready 间隔 / API-yield 间隔|{chosen.get('max_client_gap_ms')} / {chosen.get('engine_ready_gap_ms')} / {chosen.get('api_yield_gap_ms')}|",
        "",
        f"详见 `A-{variant}-analysis.json` 和 `A-{variant}-aligned-timeline.json`。事件顺序能定位长间隔发生在图文混合步及视觉 forward 之后；它支持这组负载下的时间关联，不能单独证明唯一因果机制。",
    ]
    (OUT / f"A-{variant}-analysis.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"gate": gate, "overhead": overhead, "alignment": {k: v for k, v in report["alignment"].items() if k not in ("episodes", "illustrative_max_gap_episode")}, "example": chosen}, indent=2))


if __name__ == "__main__":
    main()
