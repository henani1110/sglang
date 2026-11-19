#!/usr/bin/env python3
"""Analyze frozen ChartQA calibration predictions without touching heldout."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def med(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def question_tags(question: str) -> list[str]:
    q = question.lower()
    tags = []
    if any(term in q for term in ("value", "amount", "what is the", "what does", "denote", "indicated")):
        tags.append("value_or_axis_lookup")
    if any(term in q for term in ("highest", "lowest", "largest", "smallest", "tallest", "longest", "larger", "maximum", "minimum", "peak")):
        tags.append("extremum_or_trend")
    if any(term in q for term in ("color", "colour", "blue", "red", "green", "purple", "orange", "represent", "represented", "denoted")):
        tags.append("legend_or_visual_encoding")
    if any(term in q for term in ("how many", "how much", "number of", "count")):
        tags.append("counting")
    if any(term in q for term in ("year", "when", "between which")):
        tags.append("temporal_reading")
    if q.startswith(("is ", "are ", "does ", "do ", "did ")):
        tags.append("yes_no")
    if any(term in q for term in ("which two", "which countries", "which country", "which places")):
        tags.append("entity_identification")
    return tags or ["other"]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row.get("response_complete")]
    by_tag: dict[str, list[bool]] = {}
    for row in complete:
        for tag in question_tags(str(row.get("question", ""))):
            by_tag.setdefault(tag, []).append(bool(row.get("relaxed_accuracy_correct")))
    return {
        "rows": len(rows), "complete": len(complete), "incomplete": len(rows)-len(complete),
        "relaxed_accuracy_percent_complete": 100*statistics.mean([bool(row["relaxed_accuracy_correct"]) for row in complete]) if complete else None,
        "exact_match_percent_complete": 100*statistics.mean([bool(row["exact_match_correct"]) for row in complete]) if complete else None,
        "year_n_complete": sum(row.get("year_exact_correct") is not None for row in complete),
        "year_exact_accuracy_percent": 100*statistics.mean([bool(row["year_exact_correct"]) for row in complete if row.get("year_exact_correct") is not None]) if any(row.get("year_exact_correct") is not None for row in complete) else None,
        "median_prompt_tokens": med([float(row["prompt_tokens"]) for row in complete if row.get("prompt_tokens") is not None]),
        "median_completion_tokens": med([float(row["completion_tokens"]) for row in complete if row.get("completion_tokens") is not None]),
        "median_latency_ms": med([float(row["elapsed_client_ms"]) for row in complete if row.get("elapsed_client_ms") is not None]),
        "finish_reasons": dict(sorted(Counter(str(row.get("finish_reason")) for row in rows).items())),
        "relaxed_accuracy_by_question_tag": {
            tag: {"n": len(values), "correct": sum(values), "accuracy_percent": 100*statistics.mean(values)}
            for tag, values in sorted(by_tag.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, default=None,
                        help="Explicit mapping from edge/arm keys to attempt directories.")
    args = parser.parse_args()
    selection = json.loads(args.selection_json.read_text(encoding="utf-8")) if args.selection_json else {
        f"edge{edge}_{arm}": f"edge-{edge}/{arm}-a1" for edge in (512, 768) for arm in ("full", "k50")
    }
    cells: dict[str, list[dict[str, Any]]] = {}
    for edge in (512, 768):
        for arm in ("full", "k50"):
            key = f"edge{edge}_{arm}"
            if key not in selection:
                raise ValueError(f"missing run-selection entry for {key}")
            p = args.run_root / selection[key] / "predictions.jsonl"
            rows = read(p)
            cells[f"edge{edge}_{arm}"] = rows
    by_cell = {key: {str(row["sample_id"]): row for row in rows} for key, rows in cells.items()}
    paired: dict[str, Any] = {}
    for edge in (512, 768):
        full = by_cell[f"edge{edge}_full"]
        reduced = by_cell[f"edge{edge}_k50"]
        ids = sorted(set(full) & set(reduced))
        complete = [sample_id for sample_id in ids if full[sample_id].get("response_complete") and reduced[sample_id].get("response_complete")]
        deltas = [int(bool(reduced[sid]["relaxed_accuracy_correct"])) - int(bool(full[sid]["relaxed_accuracy_correct"])) for sid in complete]
        delta_by_id = {sid: int(bool(reduced[sid]["relaxed_accuracy_correct"])) - int(bool(full[sid]["relaxed_accuracy_correct"])) for sid in complete}
        def flip_record(sample_id: str) -> dict[str, Any]:
            source = full[sample_id]
            return {
                "sample_id": sample_id,
                "question_tags": question_tags(str(source.get("question", ""))),
                "question": source.get("question"),
                "target_answer": source.get("target_answer"),
                "full_prediction": source.get("prediction_for_scoring"),
                "k50_prediction": reduced[sample_id].get("prediction_for_scoring"),
                "full_relaxed_correct": source.get("relaxed_accuracy_correct"),
                "k50_relaxed_correct": reduced[sample_id].get("relaxed_accuracy_correct"),
                "full_year_exact": source.get("year_exact_correct"),
                "k50_year_exact": reduced[sample_id].get("year_exact_correct"),
            }
        by_image: dict[str, list[str]] = {}
        for sample_id in complete:
            image_id = str(full[sample_id].get("image_id", sample_id))
            by_image.setdefault(image_id, []).append(sample_id)
        image_deltas = {
            image_id: statistics.mean([delta_by_id[sample_id] for sample_id in sample_ids])
            for image_id, sample_ids in by_image.items()
        }
        paired[str(edge)] = {
            "paired_rows": len(ids), "paired_complete": len(complete),
            "k50_positive": sum(value > 0 for value in deltas),
            "k50_negative": sum(value < 0 for value in deltas),
            "unchanged": sum(value == 0 for value in deltas),
            "k50_positive_sample_ids": [sid for sid, value in delta_by_id.items() if value > 0],
            "k50_negative_sample_ids": [sid for sid, value in delta_by_id.items() if value < 0],
            "k50_positive_examples": [flip_record(sid) for sid, value in delta_by_id.items() if value > 0],
            "k50_negative_examples": [flip_record(sid) for sid, value in delta_by_id.items() if value < 0],
            "negative_flips_by_question_tag": {
                tag: sum(tag in question_tags(str(full[sid].get("question", ""))) for sid, value in delta_by_id.items() if value < 0)
                for tag in sorted({tag for sid, value in delta_by_id.items() if value < 0 for tag in question_tags(str(full[sid].get("question", "")))})
            },
            "image_clustered": {
                "image_clusters": len(image_deltas),
                "k50_positive_clusters": sum(value > 0 for value in image_deltas.values()),
                "k50_negative_clusters": sum(value < 0 for value in image_deltas.values()),
                "unchanged_clusters": sum(value == 0 for value in image_deltas.values()),
                "mean_cluster_accuracy_delta_k50_minus_full_points": 100*statistics.mean(image_deltas.values()) if image_deltas else None,
                "positive_cluster_deltas": {key: value for key, value in sorted(image_deltas.items()) if value > 0},
                "negative_cluster_deltas": {key: value for key, value in sorted(image_deltas.items()) if value < 0},
            },
            "mean_relaxed_accuracy_delta_k50_minus_full_points": 100*statistics.mean(deltas) if deltas else None,
            "median_prompt_token_ratio_k50_over_full": med([float(reduced[sid]["prompt_tokens"])/float(full[sid]["prompt_tokens"]) for sid in complete if reduced[sid].get("prompt_tokens") and full[sid].get("prompt_tokens")]),
            "median_latency_ratio_k50_over_full": med([float(reduced[sid]["elapsed_client_ms"])/float(full[sid]["elapsed_client_ms"]) for sid in complete if reduced[sid].get("elapsed_client_ms") and full[sid].get("elapsed_client_ms")]),
            "year_metric_flips": [sid for sid in complete if full[sid].get("year_exact_correct") != reduced[sid].get("year_exact_correct") and (full[sid].get("year_exact_correct") is not None or reduced[sid].get("year_exact_correct") is not None)],
        }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "ChartQA-H OECD-prefix local exploratory calibration subset",
        "split": "official train human-authored questions only",
        "selected_attempt_directories": selection,
        "conditions": {key: summarize(rows) for key, rows in cells.items()},
        "paired_full_vs_k50": paired,
        "scorer": "paper relaxed accuracy: parse both as float and accept <=5% relative error; otherwise case-insensitive normalized exact match",
        "diagnostics": ["Report exact match alongside relaxed accuracy.", "Four-digit year answers are also reported under strict exact matching because a 5% tolerance can admit different years."],
        "formal_quality_run": False, "eligible_quality_items_added": 0,
        "rights_status": "source_license_pending_local_exploratory_results_only",
        "heldout_used": False, "test_annotations_read": False,
        "flickr_api_or_url_used": False, "redistributable": False,
        "limitations": ["One fixed candidate subset and ordered single-pass request sequence; latency ratios are descriptive.",
                        "Image source names are inferred from filename prefixes; only a limited subset has exact source-product mapping.",
                        "Source chart data rights do not automatically establish rights to the ChartQA-rendered image."],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    notes = [
        "# ChartQA-H OECD calibration exploratory analysis",
        "",
        "This is a local, calibration-only diagnostic. It is not a formal quality run and does not add eligible items; the candidate images remain source-license-pending. Official validation heldout, test annotations, Flickr, and Statista candidates were not used.",
        "",
        "## Cell results",
        "",
        "| Condition | Complete | Relaxed accuracy | Exact match | Year exact | Median prompt tokens | Median latency (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, value in summary["conditions"].items():
        notes.append(f"| {key} | {value['complete']}/{value['rows']} | {value['relaxed_accuracy_percent_complete']} | {value['exact_match_percent_complete']} | {value['year_exact_accuracy_percent']} (n={value['year_n_complete']}) | {value['median_prompt_tokens']} | {value['median_latency_ms']} |")
    notes += ["", "## Paired comparison", "", "```json", json.dumps(paired, ensure_ascii=False, indent=2), "```", "",
              "Numeric answers use the ChartQA paper's 5% relative tolerance, while exact match and year-exact diagnostics expose scorer sensitivity. The year diagnostic is not substituted for the paper metric.",
              "", "Latency is from a single ordered pass, so it is descriptive. No threshold was fit, no heldout was touched, and the results remain exploratory while image-source/rights mapping is incomplete.", ""]
    notes_path = args.output_json.with_suffix(".md")
    notes_path.write_text("\n".join(notes), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_notes": str(notes_path), "conditions": summary["conditions"], "paired": paired}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
