#!/usr/bin/env python3
"""Check E/N replay coverage, response completeness, and synthetic image hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def local_output(block: Path, block_id: str, remote_output: str) -> Path:
    parts = PurePosixPath(remote_output).parts
    return block.joinpath(*parts[parts.index(block_id) + 1 :])


def audit(block: Path, assets: Path) -> dict:
    config = load(block / "block-config.json")
    block_id = config["block_id"]
    rows = [json.loads(line) for line in (block / "run-index.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    completion = load(block / "block-completion.json")
    failures = []
    seen_cells = Counter()
    target_arrival_counts = Counter()
    asset_hashes: dict[tuple[int, int, int, str, str], str] = {}
    hash_mismatches = []
    checked_assets = 0
    loaded_json = 0
    api_statuses = Counter()
    finish_reasons = Counter()
    response_token_counts = Counter()
    measurement_rows = []
    for index_row in rows:
        if index_row.get("status"):
            failures.append({"index_status": index_row.get("status"), "row": index_row})
            continue
        if index_row.get("returncode") != 0:
            failures.append({"nonzero_returncode": index_row.get("returncode"), "output": index_row.get("output_json")})
        is_warmup = bool(index_row.get("warmup_excluded"))
        output = local_output(block, block_id, index_row["output_json"])
        if not output.is_file():
            failures.append({"missing_output": str(output)})
            continue
        data = load(output)
        loaded_json += 1
        if not is_warmup:
            measurement_rows.append(index_row)
            cell = (index_row.get("image_size"), index_row.get("text_lanes"), index_row.get("arrival_pattern"))
            seen_cells[cell] += 1
        if data.get("image_gate_failure") is not None:
            failures.append({"image_gate_failure": data.get("image_gate_failure"), "output": str(output)})
        target_lanes = int(index_row["text_lanes"])
        texts = data.get("text_results", [])
        if len(texts) != target_lanes:
            failures.append({"text_response_count": len(texts), "expected": target_lanes, "output": str(output)})
        for response in texts:
            api_statuses[str(response.get("status"))] += 1
            finish_reasons[str(response.get("finish_reason"))] += 1
            if response.get("status") != 200 or response.get("finish_reason") != "length":
                failures.append({"text_response": {key: response.get(key) for key in ("kind", "status", "finish_reason")}, "output": str(output)})
        images = data.get("image_results", [])
        expected_images = int(index_row["image_count"])
        if len(images) != expected_images:
            failures.append({"image_response_count": len(images), "expected": expected_images, "output": str(output)})
        if not is_warmup:
            for image in images:
                lane_count = (image.get("controller_decision") or {}).get("text_lane_count")
                target_arrival_counts[str(lane_count)] += 1
                if lane_count != target_lanes:
                    failures.append({"inactive_target_lane_at_arrival": lane_count, "expected": target_lanes, "output": str(output)})
        asset_seed = "warmup" if is_warmup else f"seed-{index_row['seed']}"
        asset_cell = f"{index_row['image_size']}-{index_row['text_lanes']}lane-{index_row['arrival_pattern']}"
        for image in images:
            result = image.get("request_result", {})
            api_statuses[str(result.get("status"))] += 1
            finish_reasons[str(result.get("finish_reason"))] += 1
            usage = result.get("usage") or {}
            if usage.get("completion_tokens") is not None:
                response_token_counts[str(usage["completion_tokens"])] += 1
            if result.get("status") != 200 or result.get("finish_reason") != "length":
                failures.append({"image_response": {key: result.get(key) for key in ("kind", "status", "finish_reason", "usage")}, "output": str(output)})
            label = str(image.get("label", "")).lower()
            image_path = assets / asset_seed / asset_cell / f"joint-replay-s{index_row['seed']}-{index_row['image_size']}-{index_row['arrival_pattern']}-{label}.png"
            expected_hash = image.get("image_sha256")
            if not image_path.is_file():
                hash_mismatches.append({"missing_asset": str(image_path), "expected_hash": expected_hash})
                continue
            actual_hash = sha256(image_path)
            checked_assets += 1
            if actual_hash != expected_hash:
                hash_mismatches.append({"path": str(image_path), "expected_hash": expected_hash, "actual_hash": actual_hash})
            if not is_warmup:
                pair_key = (int(index_row["seed"]), int(index_row["image_size"]), int(index_row["text_lanes"]), index_row["arrival_pattern"], label)
                asset_hashes[pair_key] = actual_hash

    expected_seeds = set(config["seeds"])
    expected_cells = {(size, lanes, pattern) for size in (512, 768) for lanes in (2, 6) for pattern in ("uniform", "burst")}
    missing_or_extra_cells = []
    expected_pairs = {(seed, *cell) for seed in expected_seeds for cell in expected_cells}
    observed_pairs = {(row.get("seed"), row.get("image_size"), row.get("text_lanes"), row.get("arrival_pattern")) for row in measurement_rows}
    if observed_pairs != expected_pairs:
        missing_or_extra_cells = {
            "missing": sorted(expected_pairs - observed_pairs),
            "extra": sorted(observed_pairs - expected_pairs),
        }
    cross_cell_mismatches = []
    for seed in expected_seeds:
        for size in (512, 768):
            for image_index in range(1, int(config["image_count_per_cell"]) + 1):
                variants = [asset_hashes.get((seed, size, lanes, pattern, f"r{image_index}"))
                            for lanes in (2, 6) for pattern in ("uniform", "burst")]
                if len(set(variants)) > 1:
                    cross_cell_mismatches.append({"seed": seed, "image_size": size, "image_index": image_index, "hashes_by_lane_and_pattern": variants})

    assets_on_disk = sum(1 for path in assets.rglob("*.png") if path.is_file()) if assets.exists() else 0
    expected_asset_count = 4 + len(config["seeds"]) * len(expected_cells) * int(config["image_count_per_cell"])
    all_live = bool(measurement_rows) and all(target_arrival_counts.get(str(lanes), 0) == len([row for row in measurement_rows if row.get("text_lanes") == lanes]) * int(config["image_count_per_cell"]) for lanes in (2, 6)) and target_arrival_counts.get("0", 0) == 0
    return {
        "block_id": block_id,
        "formal_run": False,
        "completion": completion,
        "index_rows": len(rows),
        "loaded_output_json": loaded_json,
        "measurement_replay_count": len(measurement_rows),
        "measurement_cells": {f"{key[0]}px-{key[1]}lane-{key[2]}": value for key, value in sorted(seen_cells.items())},
        "expected_measurement_replay_count": len(config["seeds"]) * len(expected_cells),
        "missing_or_extra_seed_cells": missing_or_extra_cells,
        "http_status_counts": dict(api_statuses),
        "finish_reason_counts": dict(finish_reasons),
        "image_completion_token_counts": dict(response_token_counts),
        "text_lane_count_at_image_arrival": dict(target_arrival_counts),
        "all_target_text_lanes_active_at_every_measurement_image_arrival": all_live,
        "checked_image_assets": checked_assets,
        "expected_image_assets": expected_asset_count,
        "image_asset_files_on_disk": assets_on_disk,
        "image_hash_mismatches": hash_mismatches,
        "same_seed_same_grid_same_image_across_all_lane_and_arrival_conditions": cross_cell_mismatches,
        "failures": failures,
        "formal_run_count_added": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("block", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.block, args.assets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "block_id", "completion", "index_rows", "measurement_replay_count", "expected_measurement_replay_count",
        "missing_or_extra_seed_cells", "http_status_counts", "text_lane_count_at_image_arrival",
        "all_target_text_lanes_active_at_every_measurement_image_arrival", "checked_image_assets",
        "expected_image_assets", "image_asset_files_on_disk", "image_hash_mismatches",
        "same_seed_same_grid_same_image_across_all_lane_and_arrival_conditions", "failures", "formal_run_count_added")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
