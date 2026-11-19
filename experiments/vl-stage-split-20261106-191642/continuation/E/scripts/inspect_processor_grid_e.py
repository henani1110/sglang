#!/usr/bin/env python3
"""Report Qwen3-VL native image grids and merged visual-token counts on CPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
    )
    images = [Image.open(path).convert("RGB") for path in args.images]
    outputs = processor.image_processor(images=images, return_tensors="pt")
    grid = outputs["image_grid_thw"].tolist()
    merge_size = int(getattr(processor.image_processor, "merge_size", 2))
    size = getattr(processor.image_processor, "size", {})
    default_image_size = {
        str(key): int(value)
        for key, value in dict(size).items()
        if value is not None
    }
    items = []
    for path, row in zip(args.images, grid):
        merged_tokens = int(row[0] * row[1] * row[2] // (merge_size * merge_size))
        items.append(
            {
                "path": str(path),
                "grid_thw": [int(x) for x in row],
                "merge_size": merge_size,
                "merged_visual_tokens": merged_tokens,
            }
        )
    result = {
        "model": args.model,
        "processor_class": type(processor).__name__,
        "image_processor_class": type(processor.image_processor).__name__,
        "default_image_size": default_image_size,
        "pixel_values_shape": [int(x) for x in outputs["pixel_values"].shape],
        "image_count": len(items),
        "merged_visual_tokens_total": sum(item["merged_visual_tokens"] for item in items),
        "items": items,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
