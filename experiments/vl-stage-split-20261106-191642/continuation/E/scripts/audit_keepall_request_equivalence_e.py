#!/usr/bin/env python3
"""Audit keep-all request equivalence and trace-visible visual token counts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-jsonl", type=Path, required=True)
    parser.add_argument("--server-events", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.smoke_jsonl)
    server_rows = read_jsonl(args.server_events)
    requests = {row.get("label"): row for row in rows if row.get("response_id") is not None}
    references = {row.get("label"): row for row in rows if str(row.get("label", "")).startswith("processor-reference-")}
    errors: list[str] = []
    pairs = []
    overheads = []
    expected_visual_counts: list[int] = []
    expected_patch_rows: list[int] = []
    for size in (512, 768):
        ref = references.get(f"processor-reference-{size}")
        default = requests.get(f"default-keepall-{size}")
        explicit = requests.get(f"request-ratio-1.0-{size}")
        if ref is None or default is None or explicit is None:
            errors.append(f"missing processor reference or request pair for {size}px")
            continue
        expected = int(ref["full_decoder_tokens"])
        expected_visual_counts.extend([expected, expected])
        expected_patch_rows.extend([int(ref["full_vit_patch_rows"])] * 2)
        for row in (default, explicit):
            if row.get("status") != 200:
                errors.append(f"HTTP status is not 200 for {row.get('label')}")
            if row.get("finish_reason") != "stop":
                errors.append(f"response did not stop naturally for {row.get('label')}")
            if row.get("image_sha256") != ref.get("image_sha256"):
                errors.append(f"image SHA mismatch for {row.get('label')}")
        if default.get("content") != explicit.get("content"):
            errors.append(f"response text differs for the {size}px keep-all pair")
        default_usage = default.get("usage") or {}
        explicit_usage = explicit.get("usage") or {}
        if default_usage.get("prompt_tokens") != explicit_usage.get("prompt_tokens"):
            errors.append(f"prompt token usage differs for the {size}px keep-all pair")
        if default_usage.get("completion_tokens") != explicit_usage.get("completion_tokens"):
            errors.append(f"completion token usage differs for the {size}px keep-all pair")
        prompt_tokens = int(default_usage.get("prompt_tokens", 0))
        overheads.append(prompt_tokens - expected)
        pairs.append(
            {
                "size": size,
                "grid_thw": ref.get("image_grid_thw"),
                "full_decoder_tokens": expected,
                "full_vit_patch_rows": ref.get("full_vit_patch_rows"),
                "http_status_pair": [default.get("status"), explicit.get("status")],
                "prompt_tokens_pair": [default_usage.get("prompt_tokens"), explicit_usage.get("prompt_tokens")],
                "completion_tokens_pair": [default_usage.get("completion_tokens"), explicit_usage.get("completion_tokens")],
                "finish_reason_pair": [default.get("finish_reason"), explicit.get("finish_reason")],
                "full_response_text_equal": default.get("content") == explicit.get("content"),
                "request_ratio": 1.0,
            }
        )

    visual_events = [
        row for row in server_rows
        if row.get("kind") == "visual_forward" and row.get("request_ids")
    ]
    observed_counts = [
        int(row["output_shape"][0])
        for row in sorted(visual_events, key=lambda item: int(item.get("time_ns", 0)))
        if row.get("output_shape")
    ]
    observed_patch_rows = [
        int(row["pixel_shape"][0])
        for row in sorted(visual_events, key=lambda item: int(item.get("time_ns", 0)))
        if row.get("pixel_shape")
    ]
    if len(visual_events) != 4:
        errors.append(f"expected 4 request-linked visual_forward events, observed {len(visual_events)}")
    if observed_counts != expected_visual_counts:
        errors.append(f"visual output counts differ: expected {expected_visual_counts}, observed {observed_counts}")
    if observed_patch_rows != expected_patch_rows:
        errors.append(f"ViT patch row counts differ: expected {expected_patch_rows}, observed {observed_patch_rows}")
    if overheads and len(set(overheads)) != 1:
        errors.append(f"text/template overhead differs across grids: {overheads}")
    flush_rows = [row for row in rows if row.get("label") == "cuda-event-flush"]
    if not flush_rows or flush_rows[-1].get("status") != 200:
        errors.append("CUDA event flush did not return HTTP 200")

    result = {
        "schema_version": 1,
        "valid": not errors,
        "pair_count": len(pairs),
        "pairs": pairs,
        "text_template_overhead_tokens": overheads,
        "request_linked_visual_event_count": len(visual_events),
        "expected_visual_output_counts": expected_visual_counts,
        "observed_visual_output_counts": observed_counts,
        "expected_vit_patch_rows": expected_patch_rows,
        "observed_vit_patch_rows": observed_patch_rows,
        "interpretation": "E-patched default keep-all and request-scoped ratio=1.0 are compared on identical synthetic inputs. This does not compare against an unmodified vLLM baseline or evaluate answer quality.",
        "errors": errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"valid": result["valid"], "pairs": len(pairs), "visual_events": len(visual_events), "errors": errors}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
