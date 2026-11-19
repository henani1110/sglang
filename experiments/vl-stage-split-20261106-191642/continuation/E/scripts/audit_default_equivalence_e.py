#!/usr/bin/env python3
"""Check full-response equivalence for native and explicit processor defaults."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.smoke_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = {str(row.get("label")): row for row in rows if row.get("response_id") is not None}
    reference = next((row for row in rows if row.get("label") == "processor-reference"), None)
    errors: list[str] = []
    pairs = [
        ("native-default-768", "explicit-default-768"),
        ("native-default-512", "explicit-default-512"),
    ]
    pair_results: list[dict[str, Any]] = []
    overheads: list[int] = []
    for native_label, explicit_label in pairs:
        native = cases.get(native_label)
        explicit = cases.get(explicit_label)
        if native is None or explicit is None:
            errors.append(f"missing equivalence pair {native_label}/{explicit_label}")
            continue
        if native.get("status") != 200 or explicit.get("status") != 200:
            errors.append(f"HTTP failure in {native_label}/{explicit_label}")
        if native.get("content") != explicit.get("content"):
            errors.append(f"full response text differs in {native_label}/{explicit_label}")
        if native.get("finish_reason") != explicit.get("finish_reason"):
            errors.append(f"finish reason differs in {native_label}/{explicit_label}")
        if native.get("finish_reason") == "length" or explicit.get("finish_reason") == "length":
            errors.append(f"response ended by token limit in {native_label}/{explicit_label}")
        native_usage = native.get("usage") or {}
        explicit_usage = explicit.get("usage") or {}
        if native_usage.get("prompt_tokens") != explicit_usage.get("prompt_tokens"):
            errors.append(f"prompt token counts differ in {native_label}/{explicit_label}")
        if native_usage.get("completion_tokens") != explicit_usage.get("completion_tokens"):
            errors.append(f"completion token counts differ in {native_label}/{explicit_label}")
        if native.get("image_sha256") != explicit.get("image_sha256"):
            errors.append(f"image bytes differ in {native_label}/{explicit_label}")
        grid_item = None
        if reference is not None:
            image_name = native.get("image_name")
            grid_item = next(
                (item for item in reference.get("image_grids", []) if item.get("image_name") == image_name),
                None,
            )
        if grid_item is None:
            errors.append(f"no processor grid reference for {native_label}")
            visual_tokens = None
            overhead = None
        else:
            visual_tokens = int(grid_item["merged_visual_tokens"])
            overhead = int(native_usage.get("prompt_tokens", 0)) - visual_tokens
            overheads.append(overhead)
        pair_results.append(
            {
                "native_label": native_label,
                "explicit_label": explicit_label,
                "native_response_id": native.get("response_id"),
                "explicit_response_id": explicit.get("response_id"),
                "status_pair": [native.get("status"), explicit.get("status")],
                "prompt_tokens_pair": [native_usage.get("prompt_tokens"), explicit_usage.get("prompt_tokens")],
                "completion_tokens_pair": [native_usage.get("completion_tokens"), explicit_usage.get("completion_tokens")],
                "finish_reason_pair": [native.get("finish_reason"), explicit.get("finish_reason")],
                "full_response_text_equal": native.get("content") == explicit.get("content"),
                "merged_visual_tokens": visual_tokens,
                "text_template_overhead_tokens": overhead,
            }
        )

    if overheads and len(set(overheads)) != 1:
        errors.append(f"text/template token overhead differs across default grids: {overheads}")
    derived_overhead = overheads[0] if overheads and len(set(overheads)) == 1 else None
    cap_expectations = {"cap-262144-768": 256, "cap-65536-768": 64}
    cap_results = []
    for label, expected_visual_tokens in cap_expectations.items():
        row = cases.get(label)
        if row is None:
            errors.append(f"missing cap response {label}")
            continue
        prompt_tokens = int((row.get("usage") or {}).get("prompt_tokens", 0))
        observed_visual_tokens = prompt_tokens - derived_overhead if derived_overhead is not None else None
        if row.get("status") != 200:
            errors.append(f"HTTP failure in {label}")
        if observed_visual_tokens != expected_visual_tokens:
            errors.append(
                f"{label} expected {expected_visual_tokens} visual tokens, observed {observed_visual_tokens}"
            )
        cap_results.append(
            {
                "label": label,
                "status": row.get("status"),
                "prompt_tokens": prompt_tokens,
                "expected_visual_tokens": expected_visual_tokens,
                "observed_visual_tokens": observed_visual_tokens,
                "finish_reason": row.get("finish_reason"),
            }
        )

    result = {
        "schema_version": 1,
        "valid": not errors,
        "source_smoke_jsonl": str(args.smoke_jsonl),
        "explicit_default_size": reference.get("explicit_default_size") if reference else None,
        "processor_grid_items": reference.get("image_grids") if reference else None,
        "default_equivalence_pairs": pair_results,
        "cap_checks": cap_results,
        "errors": errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"valid": result["valid"], "pairs": len(pair_results), "caps": len(cap_results), "errors": errors}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
