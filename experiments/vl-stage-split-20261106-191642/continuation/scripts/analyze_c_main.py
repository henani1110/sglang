#!/usr/bin/env python3
"""Analyze the checkpointed C main matrix from its raw results and aligned events."""
from __future__ import annotations

import collections
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

RUN = Path(__file__).resolve().parents[2]
RAW = RUN / "continuation/raw/C-evidence"
RESULTS = RAW / "results"
LOGS = RAW / "logs"
OUT_JSON = RUN / "results/C-main-analysis.json"
OUT_MD = RUN / "results/C-main-analysis.md"
POLICIES = ("N", "F", "S", "U")
LOADS = ((512, 2), (512, 6), (768, 2), (768, 6))
BLOCKS = (1, 2, 3)
TRIALS = tuple(range(12))
SLO_GRID = ((30, 55), (25, 45), (40, 75))


def execution_order() -> dict[str, int]:
    path = RAW / "project-continuation-logs/C-run-ledger.csv"
    if not path.exists():
        return {}
    with path.open() as f:
        rows = [r for r in csv.DictReader(f) if r.get("label", "").startswith("Cmain-")]
    rows.sort(key=lambda r: r.get("start_utc", ""))
    return {r["label"]: index for index, r in enumerate(rows, 1)}


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def valid_episode(row: dict[str, Any], streams: int) -> bool:
    text = row.get("text") or []
    image = row.get("image") or {}
    return (
        not row.get("error")
        and not (row.get("validation_errors") or [])
        and len(text) == streams
        and all(x.get("token_count") == 192 and not x.get("error") for x in text)
        and image.get("token_count") == 16
        and not image.get("error")
    )


def percentile(values: Iterable[float], p: float = 0.95) -> float | None:
    data = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not data:
        return None
    x = (len(data) - 1) * p
    lo = int(x)
    hi = min(lo + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (x - lo)


def safe_mean(values: Iterable[float]) -> float | None:
    data = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return sum(data) / len(data) if data else None


def get_ids(events: list[dict[str, Any]], eid: str, streams: int) -> tuple[dict[str, int | None], dict[int, str]]:
    id_lane: dict[str, int | None] = {}
    lane_id: dict[int, str] = {}
    for ev in events:
        if ev.get("episode_id") != eid or ev.get("kind") != "sse_chunk":
            continue
        rid = ev.get("request_id")
        if not rid:
            continue
        lane = ev.get("lane")
        id_lane[str(rid)] = None if lane is None else int(lane)
        if lane is not None:
            lane_id[int(lane)] = str(rid)
    return id_lane, lane_id


def episode_metric(row: dict[str, Any], client: list[dict[str, Any]], scheduler: list[dict[str, Any]], server: list[dict[str, Any]], label: str) -> dict[str, Any]:
    eid = str(row.get("episode_id", ""))
    anchor = int(row.get("anchor_ns") or 0)
    start = int(row.get("start_ns") or anchor)
    end = int(row.get("end_ns") or 0)
    matching_client = [ev for ev in client if ev.get("episode_id") == eid]
    streams = int(row.get("text_streams") or len(row.get("text") or []))
    id_lane, lane_id = get_ids(client, eid, streams)
    lane_times: dict[int, list[int]] = collections.defaultdict(list)
    image_times: list[int] = []
    for ev in matching_client:
        if ev.get("kind") != "sse_chunk" or not (ev.get("token_ids") or []):
            continue
        t = int(ev.get("time_ns") or 0)
        lane = ev.get("lane")
        count = len(ev.get("token_ids") or [])
        if lane is None:
            image_times.extend([t] * count)
        else:
            lane_times[int(lane)].extend([t] * count)
    lo, hi = anchor - 50_000_000, anchor + 400_000_000
    gaps = []
    for times in lane_times.values():
        times.sort()
        for before, after in zip(times, times[1:]):
            left, right = max(before, lo), min(after, hi)
            if right > left:
                gaps.append((right - left) / 1e6)
    max_gap = max(gaps) if gaps else None
    ttft = ((min(image_times) - anchor) / 1e6) if image_times and anchor else None

    id_lane_sorted = sorted(id_lane.items(), key=lambda x: -len(x[0]))
    scheduler_rows = []
    for ev in scheduler:
        t = int(ev.get("time_ns") or 0)
        if start and t < start - 100_000_000:
            continue
        if end and t > end + 100_000_000:
            continue
        token_map = ev.get("num_scheduled_tokens") or {}
        enc_map = ev.get("scheduled_encoder_inputs") or {}
        matched: dict[str, Any] = {}
        for key, count in token_map.items():
            raw = next((rid for rid, _ in id_lane_sorted if key.startswith(rid)), None)
            if raw is not None:
                matched[raw] = {"lm_tokens": int(count), "encoder": False, "lane": id_lane[raw]}
        for key in enc_map:
            raw = next((rid for rid, _ in id_lane_sorted if key.startswith(rid)), None)
            if raw is not None:
                matched.setdefault(raw, {"lm_tokens": 0, "encoder": True, "lane": id_lane[raw]})
                matched[raw]["encoder"] = True
        if matched:
            scheduler_rows.append({"time_ns": t, "matched": matched, "total_tokens": ev.get("total_num_scheduled_tokens")})

    image_id = next((rid for rid, lane in id_lane_sorted if lane is None), None)
    text_ids = {lane: lane_id[lane] for lane in sorted(lane_id)}
    image_steps = [x for x in scheduler_rows if image_id and image_id in x["matched"]]
    encoder_steps = [x for x in image_steps if x["matched"][image_id].get("encoder")]
    image_p_steps = [x for x in image_steps if x["matched"][image_id].get("lm_tokens", 0) > 0]
    split_steps = [x for x in encoder_steps if x["matched"][image_id].get("lm_tokens", 0) == 0]
    same_encoder_p_step = any(
        enc["time_ns"] == prefill["time_ns"] for enc in encoder_steps for prefill in image_p_steps
    )
    d_on_split = []
    for step in split_steps:
        d_lanes = [
            lane for lane, rid in text_ids.items()
            if rid in step["matched"] and step["matched"][rid].get("lm_tokens") == 1
        ]
        d_on_split.append({"time_ns": step["time_ns"], "text_d_lanes": d_lanes, "count": len(d_lanes)})
    e_to_p = bool(split_steps and image_p_steps and split_steps[0]["time_ns"] < image_p_steps[0]["time_ns"])
    cross_step_d = bool(e_to_p and any(x["count"] > 0 for x in d_on_split))

    visual = [ev for ev in server if ev.get("kind") == "visual_forward" and start <= int(ev.get("time_ns") or 0) <= (end or 0)]
    if not end:
        visual = [ev for ev in server if ev.get("kind") == "visual_forward" and anchor <= int(ev.get("time_ns") or 0) <= anchor + 2_000_000_000]
    related_visual = []
    if encoder_steps:
        for vf in visual:
            nearest = min(encoder_steps, key=lambda x: abs(x["time_ns"] - int(vf.get("time_ns") or 0)))
            delta_ms = (int(vf.get("time_ns") or 0) - nearest["time_ns"]) / 1e6
            if abs(delta_ms) <= 100:
                related_visual.append({"time_ns": vf.get("time_ns"), "delta_from_encoder_ms": delta_ms, "output_shape": vf.get("output_shape"), "mm_hash": vf.get("mm_hash")})
    p_after_split_ns = image_p_steps[0]["time_ns"] if e_to_p else None
    enc_before_p_ns = split_steps[0]["time_ns"] if split_steps and e_to_p else None
    visual_between_e_and_p = bool(
        enc_before_p_ns is not None and p_after_split_ns is not None
        and any(enc_before_p_ns <= int(x.get("time_ns") or 0) <= p_after_split_ns for x in visual)
    )
    expected_shape = [256 if int(row.get("image_size") or 512) == 512 else 576, 8192]
    feature_shape_ok = any(x.get("output_shape") == expected_shape for x in related_visual)

    complete = valid_episode(row, streams)
    return {
        "label": label,
        "episode_id": eid,
        "strategy": row.get("arm"),
        "block": row.get("pair_block", row.get("block")),
        "attempt_id": row.get("attempt_id"),
        "trial_seed": row.get("trial_seed", row.get("trial")),
        "image_size": row.get("image_size"),
        "text_streams": streams,
        "valid": complete,
        "validation_errors": row.get("validation_errors") or [],
        "max_text_gap_ms": max_gap,
        "image_ttft_ms": ttft,
        "text_stream_gaps_in_window": len(gaps),
        "text_ids_by_lane": {str(k): v for k, v in lane_id.items()},
        "image_request_id": image_id,
        "scheduler_image_encoder_steps": len(encoder_steps),
        "scheduler_image_lm_steps": len(image_p_steps),
        "scheduler_image_encoder_only_steps": len(split_steps),
        "scheduler_encoder_and_p_same_step": same_encoder_p_step,
        "scheduler_encoder_step_d_lanes": d_on_split,
        "scheduler_e_before_p": e_to_p,
        "scheduler_during_e_before_p": cross_step_d,
        "visual_forward_between_e_and_p": visual_between_e_and_p,
        "visual_feature_shape_ok": feature_shape_ok,
        "scheduler_image_lm_token_counts": [x["matched"][image_id].get("lm_tokens", 0) for x in image_p_steps] if image_id else [],
        "visual_forward_count_in_episode_window": len(visual),
        "visual_forwards_linked_to_encoder_schedule": related_visual,
        "configured": row.get("strategy_config"),
    }


def load_all() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chosen: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}
    raw_episode_counts: collections.Counter[str] = collections.Counter()
    invalid_attempts: list[dict[str, Any]] = []
    event_cell_count = 0
    order = execution_order()
    for path in sorted(RESULTS.glob("Cmain-*.json")):
        payload = read_json(path)
        if not payload:
            continue
        m = re.match(r"Cmain-([NFSU])-L(\d+)-D(\d+)-B(\d+)-a(\d+)$", path.stem)
        if not m:
            continue
        strategy, size, streams, block, attempt = m.group(1), *map(int, m.groups()[1:])
        client_path = LOGS / f"{path.stem}-client-events.jsonl"
        sched_path = LOGS / f"{path.stem}-scheduler-events.jsonl"
        server_path = LOGS / f"{path.stem}-server-events.jsonl"
        client = read_jsonl(client_path)
        scheduler = read_jsonl(sched_path)
        server = read_jsonl(server_path)
        event_cell_count += 1
        for row in payload.get("episodes", []):
            seed = int(row.get("trial_seed", row.get("trial", -1)))
            raw_episode_counts[f"{strategy}-L{size}-D{streams}-B{block}"] += 1
            if seed not in TRIALS:
                continue
            key = (strategy, size, streams, block, seed)
            if not valid_episode(row, streams):
                invalid_attempts.append({"label": path.stem, "episode_id": row.get("episode_id"), "trial_seed": seed, "errors": row.get("validation_errors") or [row.get("error")], "strategy": strategy, "image_size": size, "text_streams": streams, "block": block})
                continue
            if key in chosen:
                continue
            metrics = episode_metric(row, client, scheduler, server, path.stem)
            metrics["run_order_rank"] = order.get(path.stem)
            chosen[key] = metrics

    expected = len(POLICIES) * len(LOADS) * len(BLOCKS) * len(TRIALS)
    missing = []
    for strategy, (size, streams), block, trial in itertools_product(POLICIES, LOADS, BLOCKS, TRIALS):
        if (strategy, size, streams, block, trial) not in chosen:
            missing.append({"strategy": strategy, "image_size": size, "text_streams": streams, "block": block, "trial_seed": trial})
    return list(chosen.values()), {
        "expected_valid_episodes": expected,
        "selected_valid_episodes": len(chosen),
        "missing_episode_keys": missing,
        "invalid_attempts": invalid_attempts,
        "raw_episode_counts_by_cell": dict(raw_episode_counts),
        "event_cells": event_cell_count,
        "execution_order_cells": len(order),
        "mean_strategy_run_order_rank": {
            arm: safe_mean([order[label] for label in order if label.startswith(f"Cmain-{arm}-")])
            for arm in POLICIES
        },
        "mean_strategy_run_order_rank_by_load": {
            f"L{size}-D{streams}": {
                arm: safe_mean([rank for label, rank in order.items() if label.startswith(f"Cmain-{arm}-L{size}-D{streams}-")])
                for arm in POLICIES
            }
            for size, streams in LOADS
        },
        "execution_order_labels": sorted(order, key=order.get),
    }


def itertools_product(*args):
    import itertools
    return itertools.product(*args)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for strategy in POLICIES:
        for size, streams in LOADS:
            key = f"{strategy}-L{size}-D{streams}"
            selected = [r for r in rows if r["strategy"] == strategy and r["image_size"] == size and r["text_streams"] == streams]
            summary[key] = {"n": len(selected), "blocks": {}}
            for block in BLOCKS:
                block_name = f"C-block{block}"
                part = [r for r in selected if r["block"] == block_name]
                sum_block = {"n": len(part)}
                for t_thr, i_thr in SLO_GRID:
                    passed = [r for r in part if r["max_text_gap_ms"] is not None and r["image_ttft_ms"] is not None and r["max_text_gap_ms"] <= t_thr and r["image_ttft_ms"] <= i_thr]
                    sum_block[f"joint_{t_thr}_{i_thr}"] = {"count": len(passed), "rate": len(passed) / len(part) if part else None}
                text = [r["max_text_gap_ms"] for r in part if r["max_text_gap_ms"] is not None]
                ttft = [r["image_ttft_ms"] for r in part if r["image_ttft_ms"] is not None]
                sum_block.update({
                    "text_gap_p50_ms": percentile(text, .50), "text_gap_p95_ms": percentile(text, .95),
                    "text_gap_max_ms": max(text) if text else None,
                    "text_gap_over_30ms": sum(x > 30 for x in text),
                    "image_ttft_p50_ms": percentile(ttft, .50), "image_ttft_p95_ms": percentile(ttft, .95),
                    "image_ttft_max_ms": max(ttft) if ttft else None,
                    "image_ttft_over_55ms": sum(x > 55 for x in ttft),
                    "valid_complete": sum(r["valid"] for r in part),
                    "e_to_p_episodes": sum(r["scheduler_e_before_p"] for r in part),
                    "s_encoder_only_with_d_episodes": sum(r["scheduler_during_e_before_p"] for r in part),
                    "e_d_p_worker_linked_episodes": sum(r["scheduler_during_e_before_p"] and r["visual_forward_between_e_and_p"] and r["visual_feature_shape_ok"] for r in part),
                    "single_visual_forward_episodes": sum(r["visual_forward_count_in_episode_window"] == 1 for r in part),
                    "visual_forward_once_with_expected_shape": sum(r["visual_forward_count_in_episode_window"] == 1 and r["visual_feature_shape_ok"] for r in part),
                    "encoder_and_p_same_step_episodes": sum(r["scheduler_encoder_and_p_same_step"] for r in part),
                })
                summary[key]["blocks"][str(block)] = sum_block
            pooled = selected
            summary[key]["pooled"] = {
                "text_gap_p50_ms": percentile([r["max_text_gap_ms"] for r in pooled if r["max_text_gap_ms"] is not None], .50),
                "text_gap_p95_ms": percentile([r["max_text_gap_ms"] for r in pooled if r["max_text_gap_ms"] is not None]),
                "image_ttft_p50_ms": percentile([r["image_ttft_ms"] for r in pooled if r["image_ttft_ms"] is not None], .50),
                "image_ttft_p95_ms": percentile([r["image_ttft_ms"] for r in pooled if r["image_ttft_ms"] is not None]),
                "joint_pass_rates": {f"{t}_{i}": sum(r["max_text_gap_ms"] is not None and r["image_ttft_ms"] is not None and r["max_text_gap_ms"] <= t and r["image_ttft_ms"] <= i for r in pooled) / len(pooled) if pooled else None for t, i in SLO_GRID},
            }

    pairs = {}
    for size, streams in LOADS:
        for block in BLOCKS:
            for other in ("F", "S", "U"):
                key = f"{other}-vs-N-L{size}-D{streams}-B{block}"
                diff_gap, diff_ttft, diff_joint = [], [], []
                n = 0
                for seed in TRIALS:
                    a = next((r for r in rows if r["strategy"] == other and r["image_size"] == size and r["text_streams"] == streams and r["block"] == f"C-block{block}" and r["trial_seed"] == seed), None)
                    b = next((r for r in rows if r["strategy"] == "N" and r["image_size"] == size and r["text_streams"] == streams and r["block"] == f"C-block{block}" and r["trial_seed"] == seed), None)
                    if not a or not b:
                        continue
                    n += 1
                    if a["max_text_gap_ms"] is not None and b["max_text_gap_ms"] is not None:
                        diff_gap.append(a["max_text_gap_ms"] - b["max_text_gap_ms"])
                    if a["image_ttft_ms"] is not None and b["image_ttft_ms"] is not None:
                        diff_ttft.append(a["image_ttft_ms"] - b["image_ttft_ms"])
                    joint_a = a["max_text_gap_ms"] is not None and a["image_ttft_ms"] is not None and a["max_text_gap_ms"] <= 30 and a["image_ttft_ms"] <= 55
                    joint_b = b["max_text_gap_ms"] is not None and b["image_ttft_ms"] is not None and b["max_text_gap_ms"] <= 30 and b["image_ttft_ms"] <= 55
                    diff_joint.append(int(joint_a) - int(joint_b))
                pairs[key] = {"n_pairs": n, "mean_delta_text_gap_ms_other_minus_N": safe_mean(diff_gap), "p95_delta_text_gap_ms_other_minus_N": percentile(diff_gap), "mean_delta_image_ttft_ms_other_minus_N": safe_mean(diff_ttft), "joint_rate_delta_other_minus_N": safe_mean(diff_joint), "joint_improved_pairs": sum(x > 0 for x in diff_joint), "joint_tied_pairs": sum(x == 0 for x in diff_joint), "joint_worse_pairs": sum(x < 0 for x in diff_joint)}
    return {"strategy_load_summary": summary, "paired_vs_N": pairs}


def solve_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    a = [matrix[i][:] + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-10:
            raise ValueError("singular order-adjustment design matrix")
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [v / scale for v in a[col]]
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col]
            if factor:
                a[r] = [u - factor * v for u, v in zip(a[r], a[col])]
    return [a[i][-1] for i in range(n)]


def order_adjusted(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive OLS sensitivity to global randomized cell order, not an inferential model."""
    categories = [f"{arm}-L{size}-D{streams}" for arm in POLICIES for size, streams in LOADS]
    category_index = {name: i for i, name in enumerate(categories)}
    blocks = ("C-block1", "C-block2", "C-block3")
    p = 1 + (len(categories) - 1) + (len(blocks) - 1) + 1

    def design(row: dict[str, Any]) -> list[float]:
        cat = f"{row['strategy']}-L{row['image_size']}-D{row['text_streams']}"
        x = [1.0]
        x.extend(1.0 if category_index[cat] == i else 0.0 for i in range(1, len(categories)))
        x.extend(1.0 if row["block"] == b else 0.0 for b in blocks[1:])
        x.append(float(row.get("run_order_rank") or 0) / 10.0)
        return x

    out: dict[str, Any] = {}
    for metric, transform in (
        ("max_text_gap_ms", lambda r: r.get("max_text_gap_ms")),
        ("image_ttft_ms", lambda r: r.get("image_ttft_ms")),
        ("joint_30_55", lambda r: int(r.get("max_text_gap_ms") is not None and r.get("image_ttft_ms") is not None and r["max_text_gap_ms"] <= 30 and r["image_ttft_ms"] <= 55)),
    ):
        observations = [(design(r), float(transform(r))) for r in rows if transform(r) is not None and r.get("run_order_rank")]
        xtx = [[0.0] * p for _ in range(p)]
        xty = [0.0] * p
        for x, y in observations:
            for i in range(p):
                xty[i] += x[i] * y
                for j in range(p):
                    xtx[i][j] += x[i] * x[j]
        beta = solve_system(xtx, xty)
        coefs = {categories[0]: 0.0}
        for i, cat in enumerate(categories[1:], 1):
            coefs[cat] = beta[i]
        order_beta = beta[-1]
        contrasts = {}
        for size, streams in LOADS:
            n = coefs[f"N-L{size}-D{streams}"]
            contrasts[f"L{size}-D{streams}"] = {
                f"{arm}_minus_N": coefs[f"{arm}-L{size}-D{streams}"] - n
                for arm in ("F", "S", "U")
            }
        out[metric] = {
            "n": len(observations),
            "linear_order_slope_per_10_cells": order_beta,
            "order_adjusted_policy_contrasts": contrasts,
            "model": "outcome ~ strategy×load fixed effects + block fixed effects + linear global cell rank; descriptive sensitivity only",
        }
    return out


def output_equivalence() -> dict[str, Any]:
    indexed: dict[tuple[int, int, int, int], dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for path in sorted(RESULTS.glob("Cmain-*.json")):
        payload = read_json(path) or {}
        m = re.match(r"Cmain-([NFSU])-L(\d+)-D(\d+)-B(\d+)-a(\d+)$", path.stem)
        if not m:
            continue
        strategy, size, streams, block = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        for row in payload.get("episodes", []):
            seed = int(row.get("trial_seed", -1))
            key = (size, streams, block, seed)
            if seed in TRIALS and valid_episode(row, streams) and strategy not in indexed[key]:
                indexed[key][strategy] = row
    out = {}
    for other in ("F", "S", "U"):
        total = exact_text = exact_image = both = 0
        mismatches = []
        by_load: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"n": 0, "exact_text": 0, "exact_image": 0, "exact_both": 0, "text_mismatch": 0, "image_mismatch": 0})
        for key, arms in indexed.items():
            if "N" not in arms or other not in arms:
                continue
            total += 1
            a, b = arms[other], arms["N"]
            load_key = f"L{key[0]}-D{key[1]}"
            text_a = [list(x.get("token_ids") or []) for x in a.get("text", [])]
            text_b = [list(x.get("token_ids") or []) for x in b.get("text", [])]
            img_a = list((a.get("image") or {}).get("token_ids") or [])
            img_b = list((b.get("image") or {}).get("token_ids") or [])
            teq, ieq = text_a == text_b, img_a == img_b
            exact_text += teq; exact_image += ieq; both += teq and ieq
            by_load[load_key]["n"] += 1
            by_load[load_key]["exact_text"] += teq
            by_load[load_key]["exact_image"] += ieq
            by_load[load_key]["exact_both"] += teq and ieq
            by_load[load_key]["text_mismatch"] += not teq
            by_load[load_key]["image_mismatch"] += not ieq
            if not teq or not ieq:
                mismatches.append({"key": list(key), "image_index": a.get("image_index"), "text_equal": teq, "image_equal": ieq, "text_first_diff_lane": first_diff(text_a, text_b), "image_first_diff": first_diff([img_a], [img_b])})
        out[f"{other}_vs_N"] = {"paired_episodes": total, "exact_text": exact_text, "exact_image": exact_image, "exact_both": both, "by_load": dict(by_load), "mismatches": mismatches}

    repeats: dict[str, Any] = {}
    for strategy in POLICIES:
        count = exact_text = exact_image = exact_both = 0
        by_load: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"n": 0, "exact_text": 0, "exact_image": 0, "exact_both": 0})
        mismatches = []
        for size, streams in LOADS:
            for seed in TRIALS:
                base = indexed.get((size, streams, 1, seed), {}).get(strategy)
                if not base:
                    continue
                for block in (2, 3):
                    row = indexed.get((size, streams, block, seed), {}).get(strategy)
                    if not row:
                        continue
                    t0 = [list(x.get("token_ids") or []) for x in base.get("text", [])]
                    t1 = [list(x.get("token_ids") or []) for x in row.get("text", [])]
                    i0 = list((base.get("image") or {}).get("token_ids") or [])
                    i1 = list((row.get("image") or {}).get("token_ids") or [])
                    teq, ieq = t0 == t1, i0 == i1
                    key = f"L{size}-D{streams}"
                    count += 1; exact_text += teq; exact_image += ieq; exact_both += teq and ieq
                    by_load[key]["n"] += 1; by_load[key]["exact_text"] += teq; by_load[key]["exact_image"] += ieq; by_load[key]["exact_both"] += teq and ieq
                    if not teq or not ieq:
                        mismatches.append({"load": key, "trial_seed": seed, "base_block": 1, "repeat_block": block, "text_equal": teq, "image_equal": ieq, "text_first_diff_lane": first_diff(t0, t1), "image_first_diff": first_diff([i0], [i1])})
        repeats[strategy] = {"repeat_pairs": count, "exact_text": exact_text, "exact_image": exact_image, "exact_both": exact_both, "by_load": dict(by_load), "mismatches": mismatches}
    return {"versus_N": out, "within_policy_repeats_vs_block1": repeats}


def first_diff(a: list[list[int]], b: list[list[int]]) -> dict[str, Any] | None:
    for lane, (x, y) in enumerate(zip(a, b)):
        if x != y:
            i = next((j for j, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
            return {"lane": lane, "index": i, "a_len": len(x), "b_len": len(y), "a_token": x[i] if i < len(x) else None, "b_token": y[i] if i < len(y) else None}
    if len(a) != len(b):
        return {"lane_count_a": len(a), "lane_count_b": len(b)}
    return None


def aligned_timeline(size: int = 768, streams: int = 6, block: int = 1, seed: int = 0) -> dict[str, Any]:
    """Create one readable N/S event timeline at the highest planned load."""
    out: dict[str, Any] = {"load": {"image_size": size, "text_streams": streams}, "block": block, "trial_seed": seed, "arms": {}}
    for strategy in ("N", "S"):
        found = None
        for path in sorted(RESULTS.glob(f"Cmain-{strategy}-L{size}-D{streams}-B{block}-a*.json")):
            payload = read_json(path) or {}
            row = next((x for x in payload.get("episodes", []) if int(x.get("trial_seed", -1)) == seed and valid_episode(x, streams)), None)
            if row:
                found = (path.stem, row)
                break
        if not found:
            continue
        label, row = found
        eid = row["episode_id"]
        client = read_jsonl(LOGS / f"{label}-client-events.jsonl")
        scheduler = read_jsonl(LOGS / f"{label}-scheduler-events.jsonl")
        server = read_jsonl(LOGS / f"{label}-server-events.jsonl")
        rel = [x for x in client if x.get("episode_id") == eid]
        anchor = int(row.get("anchor_ns") or next((x.get("time_ns", 0) for x in rel if x.get("kind") == "episode_anchor"), 0))
        start, end = int(row.get("start_ns") or anchor), int(row.get("end_ns") or 0)
        id_lane, lane_id = get_ids(client, eid, streams)
        raw_ids = sorted(id_lane.items(), key=lambda x: -len(x[0]))
        image_id = next((rid for rid, lane in raw_ids if lane is None), None)
        timeline: list[dict[str, Any]] = []

        def add(t: int, kind: str, detail: str) -> None:
            timeline.append({"time_ns": t, "relative_ms": round((t - anchor) / 1e6, 3), "kind": kind, "detail": detail})

        add(anchor, "anchor/image-send", "图像请求发出，文本流在途")
        image_sse = [x for x in rel if x.get("kind") == "sse_chunk" and x.get("lane") is None and x.get("token_ids")]
        if image_sse:
            first = min(image_sse, key=lambda x: int(x.get("time_ns") or 0))
            add(int(first["time_ns"]), "image-first-token", f"image token_ids={first.get('token_ids')}")
        for lane in range(streams):
            evs = [x for x in rel if x.get("kind") == "sse_chunk" and x.get("lane") == lane and x.get("token_ids")]
            if evs:
                first = min(evs, key=lambda x: int(x.get("time_ns") or 0))
                last = max(evs, key=lambda x: int(x.get("time_ns") or 0))
                add(int(first["time_ns"]), f"text-{lane}-first", f"request={first.get('request_id')}")
                add(int(last["time_ns"]), f"text-{lane}-last", f"token_count={row['text'][lane].get('token_count')}")

        for ev in scheduler:
            t = int(ev.get("time_ns") or 0)
            if (start and t < start - 100_000_000) or (end and t > end + 100_000_000):
                continue
            token_map = ev.get("num_scheduled_tokens") or {}
            enc_map = ev.get("scheduled_encoder_inputs") or {}
            image_key = next((k for k in list(token_map) + list(enc_map) if image_id and k.startswith(image_id)), None)
            if not image_key:
                continue
            d_lanes = []
            for lane, rid in lane_id.items():
                key = next((k for k in token_map if k.startswith(rid)), None)
                if key is not None and int(token_map[key]) == 1:
                    d_lanes.append(lane)
            add(t, "scheduler", f"image_encoder={'yes' if image_key in enc_map else 'no'}, image_LM={token_map.get(image_key, 0)}, text_D_lanes={d_lanes}, total_LM={ev.get('total_num_scheduled_tokens')}")

        visual = [x for x in server if x.get("kind") == "visual_forward" and start <= int(x.get("time_ns") or 0) <= (end or 0)]
        for ev in visual:
            add(int(ev.get("time_ns") or 0), "visual_forward", f"shape={ev.get('output_shape')}, mm_hash={ev.get('mm_hash')}")
        timeline.sort(key=lambda x: (x["time_ns"], x["kind"]))
        out["arms"][strategy] = {"label": label, "episode_id": eid, "events": timeline}
    return out


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# C 主矩阵分析",
        "",
        f"- 有效 episode：{report['coverage']['selected_valid_episodes']} / {report['coverage']['expected_valid_episodes']}。",
        f"- 配置单元：{report['coverage']['event_cells']} 个；失败尝试保留 {len(report['coverage']['invalid_attempts'])} 条；缺失配对键 {len(report['coverage']['missing_episode_keys'])} 个。",
        "- 文本间隔为 anchor−50 ms 至 anchor+400 ms 窗口内、各在途文本流相邻 token 间隔的最大值；图文 TTFT 从 anchor 到首个图文输出 token。",
        "- 联合阈值敏感性：30/55 ms（主）、25/45 ms、40/75 ms；小样本的块内 P95 属描述统计。",
        "",
        "## 各负载合并结果",
        "",
        "| 图像 | 文本路数 | 策略 | n | 文本间隔 P95 ms | 图文 TTFT P95 ms | 联合率 30/55 | 25/45 | 40/75 |",
        "|---:|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for size, streams in LOADS:
        for strategy in POLICIES:
            key = f"{strategy}-L{size}-D{streams}"
            x = report["metrics"]["strategy_load_summary"].get(key, {"n": 0, "pooled": {}})
            p = x.get("pooled", {})
            jr = p.get("joint_pass_rates", {})
            lines.append(f"| {size} | {streams} | {strategy} | {x.get('n',0)} | {fmt(p.get('text_gap_p95_ms'))} | {fmt(p.get('image_ttft_p95_ms'))} | {fmt_rate(jr.get('30_55'))} | {fmt_rate(jr.get('25_45'))} | {fmt_rate(jr.get('40_75'))} |")
    lines += ["", "## 分块结果", "", "| 图像 | 文本路数 | 块 | 策略 | n | 文本 P95 ms | TTFT P95 ms | 联合 30/55 | S encoder-only+并行 D | E→D→P 事件链 |", "|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|"]
    for size, streams in LOADS:
        for block in BLOCKS:
            for strategy in POLICIES:
                x = report["metrics"]["strategy_load_summary"].get(f"{strategy}-L{size}-D{streams}", {}).get("blocks", {}).get(str(block), {})
                joint = x.get("joint_30_55", {}).get("rate")
                lines.append(f"| {size} | {streams} | {block} | {strategy} | {x.get('n',0)} | {fmt(x.get('text_gap_p95_ms'))} | {fmt(x.get('image_ttft_p95_ms'))} | {fmt_rate(joint)} | {x.get('s_encoder_only_with_d_episodes',0)} | {x.get('e_d_p_worker_linked_episodes',0)} |")
    lines += ["", "## 输出一致性", "", "| 对照 | 配对 episode | 文本 token IDs 完全一致 | 图文 token IDs 完全一致 | 两者都一致 |", "|:---|---:|---:|---:|---:|"]
    for arm, x in report["output_equivalence"]["versus_N"].items():
        lines.append(f"| {arm} | {x['paired_episodes']} | {x['exact_text']} | {x['exact_image']} | {x['exact_both']} |")
    lines += ["", "### 策略间 token 差异位置", "", "| 对照 | 负载 | n | 文本不一致 | 图文不一致 |", "|:---|:---|---:|---:|---:|"]
    for arm, x in report["output_equivalence"]["versus_N"].items():
        for load, values in sorted(x["by_load"].items()):
            lines.append(f"| {arm} | {load} | {values['n']} | {values['text_mismatch']} | {values['image_mismatch']} |")
    lines += ["", "### 策略内跨块重复一致性", "", "| 策略 | 重复配对 | 文本 token 完全一致 | 图文 token 完全一致 | 不一致样例数 |", "|:---|---:|---:|---:|---:|"]
    for arm, values in report["output_equivalence"]["within_policy_repeats_vs_block1"].items():
        lines.append(f"| {arm} | {values['repeat_pairs']} | {values['exact_text']} | {values['exact_image']} | {len(values['mismatches'])} |")
    lines += ["", "`F−N` 的图文 token 差异只出现在 512 图两档文本并发（各 18/36），并集中于 image index 1/2；这些逐块重复在各策略内稳定。代表解码：N=`A collection of overlapping and nested geometric shapes—circles, squares, triangles,`；F=`A collection of overlapping and nested geometric shapes, including circles, squares, and triangles`。两者描述同一图像内容，但预设 16-token 上限截断输出，不能视为 token 等价或完整质量评测。`S−N` 只有 768 图/6 路文本/块2/seed0 的一个文本 token 序列不同；其余配对完全相同。该样本在重复填充的 text lane 5 尾部发生一处“be”重复。token 变化的数值根因未由当前埋点定位，不能把它写成精确输出等价。"]
    lines += ["", "## 策略相对 N 的配对差值", "", "正文本间隔/TTFT 表示该策略比 N 慢；联合率差值为该策略减 N。", "", "| 对照 | 图像 | 文本路数 | 块 | 配对数 | 平均文本间隔差 ms | 平均 TTFT 差 ms | 联合率差 | 改善/持平/变差对数 |", "|:---|---:|---:|---:|---:|---:|---:|---:|:---|"]
    for size, streams in LOADS:
        for block in BLOCKS:
            for arm in ("F", "S", "U"):
                x = report["metrics"]["paired_vs_N"].get(f"{arm}-vs-N-L{size}-D{streams}-B{block}", {})
                lines.append(f"| {arm}−N | {size} | {streams} | {block} | {x.get('n_pairs',0)} | {fmt(x.get('mean_delta_text_gap_ms_other_minus_N'))} | {fmt(x.get('mean_delta_image_ttft_ms_other_minus_N'))} | {fmt_rate(x.get('joint_rate_delta_other_minus_N'))} | {x.get('joint_improved_pairs',0)}/{x.get('joint_tied_pairs',0)}/{x.get('joint_worse_pairs',0)} |")
    lines += ["", "## 执行顺序敏感性", "", f"48 个策略单元的 runner ledger 全部 ready 且客户端退出码为 0。实际执行顺序是全局固定种子的随机置换，不是方案中原定的负载/块内均衡顺序；平均名次为 N={report['coverage']['mean_strategy_run_order_rank']['N']:.2f}, F={report['coverage']['mean_strategy_run_order_rank']['F']:.2f}, S={report['coverage']['mean_strategy_run_order_rank']['S']:.2f}, U={report['coverage']['mean_strategy_run_order_rank']['U']:.2f}（1 最早、48 最晚）。下表给出各负载内的名次均值；另用线性顺序项作描述性敏感性，随机顺序造成的未均衡仍作为局限。", "", "| 负载 | N 名次均值 | F 名次均值 | S 名次均值 | U 名次均值 |", "|:---|---:|---:|---:|---:|"]
    for load, ranks in report["coverage"]["mean_strategy_run_order_rank_by_load"].items():
        lines.append(f"| {load} | {ranks['N']:.2f} | {ranks['F']:.2f} | {ranks['S']:.2f} | {ranks['U']:.2f} |")
    lines += ["", "| 结果变量 | 每晚 10 个单元变化 | F−N L512/D2 | S−N L768/D2 | S−N L768/D6 |", "|:---|---:|---:|---:|---:|"]
    sens = report["order_adjusted_sensitivity"]
    for metric, label in (("max_text_gap_ms", "文本间隔 ms"), ("image_ttft_ms", "图文 TTFT ms"), ("joint_30_55", "联合通过比例")):
        x = sens[metric]
        c1 = x["order_adjusted_policy_contrasts"]["L512-D2"]["F_minus_N"]
        c2 = x["order_adjusted_policy_contrasts"]["L768-D2"]["S_minus_N"]
        c3 = x["order_adjusted_policy_contrasts"]["L768-D6"]["S_minus_N"]
        lines.append(f"| {label} | {fmt(x['linear_order_slope_per_10_cells'])} | {fmt(c1)} | {fmt(c2)} | {fmt(c3)} |")
    lines += ["", "模型为 `outcome ~ strategy×load 固定效应 + block 固定效应 + 全局单元名次线性项`，只作随机化不均衡的描述性敏感性，不产生显著性结论。"]
    lines += ["", "## 调度机制和证据覆盖", "", "各 episode 的 scheduler/worker 事件细节见 JSON 的 `episode_metrics`。"]
    lines += ["", "| 策略 | episode | E 与图文 P 同 scheduler step | 拆分 E step 同步推进文本 D、之后 P、期间有 worker E | 一次 E 且 feature shape 正确 |", "|:---:|---:|---:|---:|---:|"]
    all_episodes = report["episode_metrics"]
    for arm in POLICIES:
        part = [r for r in all_episodes if r["strategy"] == arm]
        split_chain = sum(r["scheduler_during_e_before_p"] and r["visual_forward_between_e_and_p"] and r["visual_feature_shape_ok"] for r in part)
        shape_once = sum(r["visual_forward_count_in_episode_window"] == 1 and r["visual_feature_shape_ok"] for r in part)
        lines.append(f"| {arm} | {len(part)} | {sum(r['scheduler_encoder_and_p_same_step'] for r in part)} | {split_chain} | {shape_once} |")
    lines += ["", "S 的拆步有效路径按 episode 计，需同时满足 image encoder-only scheduler step、该 step 推进在途文本 D、后续 image P、worker visual forward 位于 E/P 之间及预期输出形状；不能仅凭客户端完成或调度延迟推断。"]
    lines += ["", "## 最高负载 N/S 对齐时间线（768 图、6 路文本、块 1、seed 0）", "", "| 策略 | 相对 anchor ms | 事件 | 详情 |", "|:---:|---:|:---|:---|"]
    for strategy in ("N", "S"):
        arm = report.get("aligned_timeline", {}).get("arms", {}).get(strategy, {})
        for event in arm.get("events", []):
            lines.append(f"| {strategy} | {event['relative_ms']:.3f} | {event['kind']} | {event['detail']} |")
    lines += ["", "## 解释边界", "", "这是封闭回放与固定到达轨迹上的系统实验，不等于开放环服务的 goodput/线上 SLO。每单元 12 个 episode、每负载 3 个块；结论按描述性差异与块方向解释，不作总体人群推断。失败尝试和缺失配对键单列，不进入有效 n。", ""]
    return "\n".join(lines)


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.3f}"


def fmt_rate(value: Any) -> str:
    return "NA" if value is None else f"{100*float(value):.1f}%"


def main() -> None:
    rows, coverage = load_all()
    report = {
        "method": "C fixed replay; client SSE window-gap / image TTFT, scheduler request IDs matched to per-episode SSE IDs, worker visual-forward correlated by same cell time window.",
        "slo_thresholds_ms": {"primary": {"text_gap": 30, "image_ttft": 55}, "sensitivity": [{"text_gap": t, "image_ttft": i} for t, i in SLO_GRID[1:]]},
        "coverage": coverage,
        "metrics": summarize(rows),
        "output_equivalence": output_equivalence(),
        "order_adjusted_sensitivity": order_adjusted(rows),
        "aligned_timeline": aligned_timeline(),
        "episode_metrics": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    OUT_MD.write_text(markdown(report))
    print(json.dumps({"valid": coverage["selected_valid_episodes"], "expected": coverage["expected_valid_episodes"], "missing": len(coverage["missing_episode_keys"]), "invalid_attempts": len(coverage["invalid_attempts"]), "markdown": str(OUT_MD), "json": str(OUT_JSON)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
