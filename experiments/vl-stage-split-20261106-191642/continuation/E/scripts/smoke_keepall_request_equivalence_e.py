#!/usr/bin/env python3
"""Compare E-patched default keep-all to explicit request-scoped ratio 1.0."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor

from smoke_batch_multi_image_e import make_chart
from smoke_default_equivalence_e import run_request
from smoke_batch_endpoint_image_pruning_e import flush_events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8285/v1/chat/completions")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--asset-dir", required=True)
    args = parser.parse_args()

    model = Path(args.model)
    asset_dir = Path(args.asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    records: list[dict] = []
    prompt = "Read the title of this chart and answer in one short sentence."
    for size, color in ((512, (35, 150, 95)), (768, (35, 100, 185))):
        payload = make_chart(f"KEEPALL-{size}", (size, size), color)
        image_name = f"square-{size}"
        asset_path = asset_dir / f"{image_name}.png"
        asset_path.write_bytes(payload)
        image_hash = hashlib.sha256(payload).hexdigest()
        image = Image.open(asset_path).convert("RGB")
        processed = processor.image_processor(images=[image], return_tensors="pt")
        grid = [int(x) for x in processed["image_grid_thw"][0].tolist()]
        merge_size = int(getattr(processor.image_processor, "merge_size", 2))
        full_tokens = grid[0] * grid[1] * grid[2] // (merge_size * merge_size)
        records.append(
            {
                "label": f"processor-reference-{size}",
                "image_name": image_name,
                "image_sha256": image_hash,
                "image_grid_thw": grid,
                "full_decoder_tokens": full_tokens,
                "full_vit_patch_rows": int(processed["pixel_values"].shape[0]),
            }
        )
        records.append(
            run_request(args.base_url, f"default-keepall-{size}", payload, image_name, image_hash, prompt, None)
        )
        records.append(
            run_request(
                args.base_url,
                f"request-ratio-1.0-{size}",
                payload,
                image_name,
                image_hash,
                prompt,
                {"images_kwargs": {"e_keep_ratio": 1.0}},
            )
        )
    records.append(flush_events(args.base_url))

    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"records": len(records), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
