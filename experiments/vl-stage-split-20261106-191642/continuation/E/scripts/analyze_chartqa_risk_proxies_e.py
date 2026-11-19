#!/usr/bin/env python3
"""Describe whether cheap query/image features track ChartQA budget sensitivity.

This is a calibration-only association report. It does not fit thresholds,
claim risk calibration, or activate the serving controller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = ((start + 1) + end) / 2.0
        for pos in range(start, end):
            ranks[order[pos]] = rank
        start = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = average_ranks(x), average_ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    dx = [value - mx for value in rx]
    dy = [value - my for value in ry]
    denom = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return None if denom == 0 else sum(a * b for a, b in zip(dx, dy)) / denom


def image_features(path: Path) -> dict[str, float]:
    with Image.open(path) as image:
        gray = image.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    hist, _ = np.histogram(arr, bins=32, range=(0.0, 1.0), density=False)
    probs = hist.astype(np.float64) / max(1, int(hist.sum()))
    nonzero = probs[probs > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())
    edges = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    return {
        "gray_mean": float(arr.mean()),
        "gray_std": float(arr.std()),
        "gray_variance": float(arr.var()),
        "histogram_entropy_bits_32bin": entropy,
        "edge_mean": float(edges.mean()),
        "edge_fraction_gt_0_08": float((edges > 0.08).mean()),
        "dark_fraction_lt_0_35": float((arr < 0.35).mean()),
        "light_fraction_gt_0_90": float((arr > 0.90).mean()),
    }


def query_features(question: str) -> dict[str, float]:
    q = question.casefold()
    words = [word for word in q.replace("?", " ").replace(",", " ").split() if word]
    ratio_terms = ("ratio", "percent", "percentage", "proportion", "times", "difference", "sum", "average")
    compare_terms = ("higher", "lower", "highest", "lowest", "more", "less", "compare", "largest", "smallest")
    temporal_terms = ("year", "years", "between", "from", "to", "before", "after", "trend")
    entity_terms = ("which", "country", "countries", "place", "places", "group", "category")
    return {
        "question_word_count": float(len(words)),
        "question_char_count": float(len(question)),
        "has_numeric_literal": float(any(char.isdigit() for char in question)),
        "ratio_or_arithmetic_terms": float(sum(term in q for term in ratio_terms)),
        "comparison_terms": float(sum(term in q for term in compare_terms)),
        "temporal_terms": float(sum(term in q for term in temporal_terms)),
        "entity_lookup_terms": float(sum(term in q for term in entity_terms)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection_json.read_text(encoding="utf-8"))
    cells: dict[str, dict[str, dict[str, Any]]] = {}
    for key in ("edge512_full", "edge512_k25", "edge512_k50", "edge512_k75",
                "edge768_full", "edge768_k25", "edge768_k50", "edge768_k75"):
        relative = selection.get(key)
        if not relative:
            raise ValueError(f"missing selected arm: {key}")
        rows = read_jsonl(args.run_root / relative / "predictions.jsonl")
        cells[key] = {str(row["sample_id"]): row for row in rows}

    calibration_rows = list(cells["edge512_full"].values())
    if any(row.get("quality_split") != "calibration" for row in calibration_rows):
        raise ValueError("refusing non-calibration prediction rows")
    if any(row.get("source_split") != "train" for row in calibration_rows):
        raise ValueError("refusing non-train prediction rows")

    per_image: dict[str, dict[str, Any]] = {}
    for row in calibration_rows:
        image_id = str(row["image_id"])
        item = per_image.setdefault(image_id, {"questions": [], "features": None})
        item["questions"].append({"sample_id": row["sample_id"], "question": row["question"]})
        if item["features"] is None:
            path = args.images_root / "images" / "train" / image_id
            if not path.is_file():
                raise FileNotFoundError(path)
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != row.get("image_sha256"):
                raise ValueError(f"image hash mismatch for {image_id}")
            item["features"] = image_features(path)

    results: dict[str, Any] = {}
    for edge in (512, 768):
        full = cells[f"edge{edge}_full"]
        for arm in ("k25", "k50", "k75"):
            compressed = cells[f"edge{edge}_{arm}"]
            image_rows: dict[str, list[dict[str, Any]]] = {}
            paired = 0
            lost = gained = unchanged = 0
            for sample_id in sorted(set(full) & set(compressed)):
                base, reduced = full[sample_id], compressed[sample_id]
                if not base.get("response_complete") or not reduced.get("response_complete"):
                    continue
                paired += 1
                base_ok = bool(base.get("relaxed_accuracy_correct"))
                reduced_ok = bool(reduced.get("relaxed_accuracy_correct"))
                lost += int(base_ok and not reduced_ok)
                gained += int(reduced_ok and not base_ok)
                unchanged += int(base_ok == reduced_ok)
                image_rows.setdefault(str(base["image_id"]), []).append(
                    {"sample_id": sample_id, "delta": int(base_ok) - int(reduced_ok)}
                )

            cluster_records: list[dict[str, Any]] = []
            feature_names: set[str] = set()
            for image_id, rows in sorted(image_rows.items()):
                item = per_image[image_id]
                feat = dict(item["features"])
                qrows = [query_features(entry["question"]) for entry in item["questions"]]
                for name in qrows[0] if qrows else []:
                    feat[f"query_mean_{name}"] = statistics.mean(qrow[name] for qrow in qrows)
                feat["question_count"] = float(len(item["questions"]))
                feature_names.update(feat)
                cluster_records.append({
                    "image_id": image_id,
                    "complete_paired_questions": len(rows),
                    "mean_accuracy_loss_full_minus_budget": statistics.mean(row["delta"] for row in rows),
                    "lost_correct_questions": sum(row["delta"] > 0 for row in rows),
                    "gained_correct_questions": sum(row["delta"] < 0 for row in rows),
                    "features": feat,
                })

            associations: dict[str, Any] = {}
            for name in sorted(feature_names):
                x = [float(row["features"][name]) for row in cluster_records]
                y = [float(row["mean_accuracy_loss_full_minus_budget"]) for row in cluster_records]
                associations[name] = {
                    "spearman_rho": spearman(x, y),
                    "n_image_clusters": len(cluster_records),
                }
            results[f"edge{edge}_{arm}"] = {
                "paired_complete_questions": paired,
                "lost_correct_questions": lost,
                "gained_correct_questions": gained,
                "unchanged_correctness_questions": unchanged,
                "image_clusters": len(cluster_records),
                "cluster_level_feature_association_with_accuracy_loss": associations,
                "image_cluster_records": cluster_records,
            }

    output = {
        "analysis_id": "chartqa-calibration-risk-proxy-association-v0.1",
        "dataset": "ChartQA-H human-authored train calibration; OECD filename-prefix subset",
        "candidate_rows": len(calibration_rows),
        "unique_images": len(per_image),
        "test_annotations_read": False,
        "heldout_used": False,
        "flickr_api_or_url_used": False,
        "formal_quality_run": False,
        "rights_status": "source_license_pending_local_exploratory_evaluation_only",
        "feature_scope": "transparent question lexical counts and resized grayscale/edge statistics; neither SieveServe QCR/DIG nor a calibrated safety signal",
        "method": "descriptive per-image Spearman association between pre-E features and full-minus-budget relaxed-accuracy delta; no fitting, threshold selection, or causal claim",
        "conditions": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: {"paired": value["paired_complete_questions"],
                            "lost": value["lost_correct_questions"],
                            "gained": value["gained_correct_questions"],
                            "clusters": value["image_clusters"]}
                      for key, value in results.items()}, ensure_ascii=False, indent=2))
    print(f"output={args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
