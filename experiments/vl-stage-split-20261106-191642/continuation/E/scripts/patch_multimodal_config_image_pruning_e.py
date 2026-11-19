#!/usr/bin/env python3
"""Enable vLLM's existing M-RoPE pruning bridge for E image pruning.

The E Qwen3-VL image path carries selected coordinates in the established
five-channel envelope. vLLM's model-state pruner must be enabled so it can
strip those channels and update M-RoPE positions before DeepStack embedding
merge. This patch is isolated to the E overlay and is gated by
VL_IMAGE_KEEP_RATIO.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path

EXPECTED_SOURCE_SHA256 = "4a232a4a42af15846251b86d6d3655cea34cf350cb135ec83b4dbd23b44b4741"
LOGICAL_SOURCE = "vllm/config/multimodal.py"
LOGICAL_OUTPUT = "overlay/vllm/config/multimodal.py"
LOGICAL_SNAPSHOT = "source_snapshot/multimodal_config_before_image_pruning_e.py"
LOGICAL_DIFF = "patches/multimodal_config_image_pruning_e.diff"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, got {count}")
    return text.replace(old, new, 1)


def patch_source(source: str) -> str:
    source = replace_once(
        source,
        "import torch\n",
        "import os\n\nimport torch\n",
        "stdlib import",
    )
    source = replace_once(
        source,
        "    def is_multimodal_pruning_enabled(self):\n"
        "        return self.get_video_pruning_spec() is not None\n",
        "    def is_multimodal_pruning_enabled(self):\n"
        "        return (\n"
        "            self.get_video_pruning_spec() is not None\n"
        "            or os.environ.get(\"VL_IMAGE_KEEP_RATIO\") is not None\n"
        "        )\n",
        "E image pruning gate",
    )
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diff", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    source_sha = sha256(source_bytes)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"unexpected multimodal.py SHA-256 {source_sha}; "
            f"expected {EXPECTED_SOURCE_SHA256}"
        )
    source = source_bytes.decode("utf-8")
    if "VL_IMAGE_KEEP_RATIO\") is not None" in source:
        raise RuntimeError("E image-pruning gate already exists in source")
    patched = patch_source(source)
    patched_bytes = patched.encode("utf-8")

    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot.exists():
        if sha256(args.snapshot.read_bytes()) != EXPECTED_SOURCE_SHA256:
            raise RuntimeError(f"snapshot exists with unexpected content: {args.snapshot}")
    else:
        args.snapshot.write_bytes(source_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.diff.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(patched_bytes)
    diff_text = "".join(
        difflib.unified_diff(
            source.splitlines(True),
            patched.splitlines(True),
            fromfile=f"a/{LOGICAL_SOURCE}",
            tofile=f"b/{LOGICAL_OUTPUT}",
        )
    )
    args.diff.write_text(diff_text, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "mode": "E-image-pruning-M-RoPE-config-gate",
        "source_path": LOGICAL_SOURCE,
        "source_sha256": source_sha,
        "output_path": LOGICAL_OUTPUT,
        "output_sha256": sha256(patched_bytes),
        "snapshot_path": LOGICAL_SNAPSHOT,
        "diff_path": LOGICAL_DIFF,
        "diff_sha256": sha256(diff_text.encode("utf-8")),
        "gate": "MultiModalConfig.is_multimodal_pruning_enabled returns true when VL_IMAGE_KEEP_RATIO is set, activating vLLM's existing multimodal M-RoPE strip/recompute bridge",
        "scope": "E-only overlay; leaves installed vLLM source unchanged",
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
