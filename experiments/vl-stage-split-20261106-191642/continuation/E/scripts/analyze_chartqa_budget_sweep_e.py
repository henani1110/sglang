#!/usr/bin/env python3
"""Analyze frozen ChartQA calibration runs over multiple visual keep ratios."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARMS = ("full", "k25", "k50", "k75")


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def med(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def question_tags(question: str) -> list[str]:
    q = question.lower()
    tags = []
    if any(t in q for t in ("value", "amount", "what is the", "what does", "denote", "indicated")):
        tags.append("value_or_axis_lookup")
    if any(t in q for t in ("highest", "lowest", "largest", "smallest", "tallest", "longest", "larger", "maximum", "minimum", "peak")):
        tags.append("extremum_or_trend")
    if any(t in q for t in ("color", "colour", "blue", "red", "green", "purple", "orange", "represent", "represented", "denoted")):
        tags.append("legend_or_visual_encoding")
    if any(t in q for t in ("how many", "how much", "number of", "count")):
        tags.append("counting")
    if any(t in q for t in ("year", "when", "between which")):
        tags.append("temporal_reading")
    if q.startswith(("is ", "are ", "does ", "do ", "did ")):
        tags.append("yes_no")
    if any(t in q for t in ("which two", "which countries", "which country", "which places")):
        tags.append("entity_identification")
    return tags or ["other"]


def cell_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row.get("response_complete")]
    tagged: dict[str, list[bool]] = {}
    for row in complete:
        for tag in question_tags(str(row.get("question", ""))):
            tagged.setdefault(tag, []).append(bool(row.get("relaxed_accuracy_correct")))
    years = [bool(row["year_exact_correct"]) for row in complete if row.get("year_exact_correct") is not None]
    return {
        "rows": len(rows), "complete": len(complete), "incomplete": len(rows)-len(complete),
        "relaxed_accuracy_percent_complete": 100*statistics.mean([bool(row["relaxed_accuracy_correct"]) for row in complete]) if complete else None,
        "exact_match_percent_complete": 100*statistics.mean([bool(row["exact_match_correct"]) for row in complete]) if complete else None,
        "year_n_complete": len(years),
        "year_exact_accuracy_percent": 100*statistics.mean(years) if years else None,
        "median_prompt_tokens": med([float(row["prompt_tokens"]) for row in complete if row.get("prompt_tokens") is not None]),
        "median_completion_tokens": med([float(row["completion_tokens"]) for row in complete if row.get("completion_tokens") is not None]),
        "median_latency_ms": med([float(row["elapsed_client_ms"]) for row in complete if row.get("elapsed_client_ms") is not None]),
        "finish_reasons": dict(sorted(Counter(str(row.get("finish_reason")) for row in rows).items())),
        "relaxed_accuracy_by_question_tag": {
            tag: {"n": len(values), "correct": sum(values), "accuracy_percent": 100*statistics.mean(values)}
            for tag, values in sorted(tagged.items())
        },
    }


def pair_summary(full: dict[str, dict[str, Any]], compressed: dict[str, dict[str, Any]], edge: int, arm: str) -> dict[str, Any]:
    ids = sorted(set(full) & set(compressed))
    paired = [sid for sid in ids if full[sid].get("response_complete") and compressed[sid].get("response_complete")]
    delta = {sid: int(bool(compressed[sid].get("relaxed_accuracy_correct")))
             - int(bool(full[sid].get("relaxed_accuracy_correct"))) for sid in paired}
    positive = [sid for sid, value in delta.items() if value > 0]
    negative = [sid for sid, value in delta.items() if value < 0]
    by_image: dict[str, list[int]] = {}
    for sid in paired:
        image_id = str(full[sid].get("image_id", sid))
        by_image.setdefault(image_id, []).append(delta[sid])
    image_delta = {image: statistics.mean(values) for image, values in by_image.items()}

    def example(sid: str) -> dict[str, Any]:
        return {
            "sample_id": sid,
            "image_id": full[sid].get("image_id"),
            "question_tags": question_tags(str(full[sid].get("question", ""))),
            "question": full[sid].get("question"),
            "target_answer": full[sid].get("target_answer"),
            "full_prediction": full[sid].get("prediction_for_scoring"),
            "compressed_prediction": compressed[sid].get("prediction_for_scoring"),
            "full_correct": full[sid].get("relaxed_accuracy_correct"),
            "compressed_correct": compressed[sid].get("relaxed_accuracy_correct"),
            "full_year_exact": full[sid].get("year_exact_correct"),
            "compressed_year_exact": compressed[sid].get("year_exact_correct"),
        }

    negative_tags = sorted({tag for sid in negative for tag in question_tags(str(full[sid].get("question", "")))})
    return {
        "edge": edge, "compressed_arm": arm, "paired_rows": len(ids), "paired_complete": len(paired),
        "positive": len(positive), "negative": len(negative), "unchanged": sum(value == 0 for value in delta.values()),
        "positive_sample_ids": positive, "negative_sample_ids": negative,
        "positive_examples": [example(sid) for sid in positive],
        "negative_examples": [example(sid) for sid in negative],
        "negative_flips_by_question_tag": {
            tag: sum(tag in question_tags(str(full[sid].get("question", ""))) for sid in negative)
            for tag in negative_tags
        },
        "image_clustered": {
            "image_clusters": len(image_delta),
            "positive_clusters": sum(value > 0 for value in image_delta.values()),
            "negative_clusters": sum(value < 0 for value in image_delta.values()),
            "unchanged_clusters": sum(value == 0 for value in image_delta.values()),
            "mean_cluster_accuracy_delta_points": 100*statistics.mean(image_delta.values()) if image_delta else None,
            "positive_cluster_deltas": {k: v for k, v in sorted(image_delta.items()) if v > 0},
            "negative_cluster_deltas": {k: v for k, v in sorted(image_delta.items()) if v < 0},
        },
        "mean_accuracy_delta_points": 100*statistics.mean(delta.values()) if delta else None,
        "median_prompt_token_ratio_over_full": med([
            float(compressed[sid]["prompt_tokens"])/float(full[sid]["prompt_tokens"])
            for sid in paired if compressed[sid].get("prompt_tokens") and full[sid].get("prompt_tokens")]),
        "median_latency_ratio_over_full": med([
            float(compressed[sid]["elapsed_client_ms"])/float(full[sid]["elapsed_client_ms"])
            for sid in paired if compressed[sid].get("elapsed_client_ms") and full[sid].get("elapsed_client_ms")]),
        "year_metric_flips": [sid for sid in paired
                              if full[sid].get("year_exact_correct") != compressed[sid].get("year_exact_correct")
                              and (full[sid].get("year_exact_correct") is not None or compressed[sid].get("year_exact_correct") is not None)],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection_json.read_text(encoding="utf-8"))
    cells: dict[str, list[dict[str, Any]]] = {}
    for edge in (512, 768):
        for arm in ARMS:
            key = f"edge{edge}_{arm}"
            if key not in selection:
                continue
            cells[key] = read(args.run_root / selection[key] / "predictions.jsonl")
    summaries = {key: cell_summary(rows) for key, rows in cells.items()}
    indexed = {key: {str(row["sample_id"]): row for row in rows} for key, rows in cells.items()}
    paired: dict[str, dict[str, Any]] = {}
    for edge in (512, 768):
        full_key = f"edge{edge}_full"
        if full_key not in indexed:
            raise ValueError(f"missing full-token condition {full_key}")
        paired[str(edge)] = {}
        for arm in ("k25", "k50", "k75"):
            key = f"edge{edge}_{arm}"
            if key in indexed:
                paired[str(edge)][arm] = pair_summary(indexed[full_key], indexed[key], edge, arm)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "ChartQA-H OECD-prefix local exploratory calibration subset",
        "split": "official train human-authored questions only",
        "selected_attempt_directories": selection,
        "conditions": summaries,
        "paired_full_vs_compressed": paired,
        "scorer": "paper relaxed accuracy: parse both as float and accept <=5% relative error; otherwise case-insensitive normalized exact match",
        "diagnostics": ["strict normalized exact match", "year-like targets strict exact match; paper metric remains unchanged"],
        "formal_quality_run": False, "eligible_quality_items_added": 0,
        "rights_status": "source_license_pending_local_exploratory_results_only",
        "heldout_used": False, "test_annotations_read": False,
        "flickr_api_or_url_used": False, "redistributable": False,
        "limitations": ["40 unique train chart images; question rows within one chart are clustered.",
                        "One fixed candidate subset and ordered single-pass request sequence; latency ratios are descriptive.",
                        "Image source names are inferred from filename prefixes; only a limited subset has exact source-product mapping.",
                        "Source chart data rights do not automatically establish rights to the ChartQA-rendered image."],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    notes = [
        "# ChartQA-H OECD calibration budget sweep",
        "",
        "This is a local calibration-only diagnostic, not a formal quality run. It adds no eligible items. All image-source rights remain pending; official validation heldout, test annotations, Flickr, and Statista candidates were not used.",
        "",
        "## Condition summary",
        "",
        "| Condition | Complete | Relaxed accuracy | Exact match | Year exact | Median prompt tokens | Median latency (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, value in summaries.items():
        relaxed = value["relaxed_accuracy_percent_complete"]
        exact = value["exact_match_percent_complete"]
        year = value["year_exact_accuracy_percent"]
        notes.append(
            f"| {key} | {value['complete']}/{value['rows']} | "
            f"{relaxed:.2f} | {exact:.2f} | {year:.2f} (n={value['year_n_complete']}) | "
            f"{value['median_prompt_tokens']:.1f} | {value['median_latency_ms']:.2f} |"
        )
    notes += ["", "## Paired results against full tokens", "",
              "| Edge | Keep | Paired | Better | Worse | Image clusters worse | Accuracy delta (pp) | Median token ratio | Median latency ratio |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for edge, arms in paired.items():
        for arm, value in arms.items():
            cluster = value["image_clustered"]
            notes.append(
                f"| {edge} | {arm} | {value['paired_complete']} | {value['positive']} | "
                f"{value['negative']} | {cluster['negative_clusters']} | "
                f"{value['mean_accuracy_delta_points']:.2f} | "
                f"{value['median_prompt_token_ratio_over_full']:.4f} | "
                f"{value['median_latency_ratio_over_full']:.4f} |"
            )
    notes += ["", "Question-tag counts overlap because a question can receive multiple rule-based tags; they are descriptive proxies, not calibrated QCR/DIG or a safety threshold.",
              "Year-exact results are shown as a sensitivity diagnostic alongside the paper-style 5% metric. Latencies are descriptive from a single ordered pass.",
              "", "Heldout remains frozen and unused. Do not treat these 80 calibration questions or 40 chart clusters as a general quality-preservation estimate.", ""]
    notes_path = args.output_json.with_suffix(".md")
    notes_path.write_text("\n".join(notes), encoding="utf-8")
    compact = {"output_json": str(args.output_json), "output_notes": str(notes_path),
               "conditions": {key: {k: value[k] for k in ("complete", "relaxed_accuracy_percent_complete", "exact_match_percent_complete", "year_exact_accuracy_percent", "median_prompt_tokens", "median_latency_ms")}
                              for key, value in summaries.items()},
               "paired": {edge: {arm: {k: value[k] for k in ("paired_complete", "positive", "negative", "mean_accuracy_delta_points", "median_prompt_token_ratio_over_full", "median_latency_ratio_over_full", "image_clustered")}
                                  for arm, value in arms.items()} for edge, arms in paired.items()}}
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
