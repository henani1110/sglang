#!/usr/bin/env python3
"""Measure CUDA stream time at vLLM's actual multimodal tensor H2D copy."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path

EXPECTED_SOURCE_SHA256 = "3ce5d3cc5bedf0343d75e1a7cfbca10a0061b960de56a9fb6465cde0f14bc8b9"
IMPORT_ANCHOR = "import numpy as np\n"
IMPORT_REPLACEMENT = IMPORT_ANCHOR + "import os\n\n"
HOOK_IMPORT_ANCHOR = "from vllm.utils.jsontree import json_iter_leaves, json_map_leaves\n"
HOOK_IMPORT_REPLACEMENT = HOOK_IMPORT_ANCHOR + "from vllm.vl_pilot_hook import (\n    current_mm_keys,\n    mm_request_ids_for_hashes,\n    record_stage_cuda_event,\n)\n"
FUNCTION_ANCHOR = """def _nested_tensors_h2d(
    tensors: NestedTensors,
    device: torch.types.Device,
    *,
    pin_memory: bool = False,
) -> NestedTensors:
    if device is None:
        return tensors

    def _h2d(x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            return x
        if pin_memory and x.is_cpu and not (x.is_pinned() and _is_dense(x)):
            # Ensure tensor is pinned and dense for non_blocking H2D copy.
            x = x.new_empty(x.shape, pin_memory=True).copy_(x)
        return x.to(device=device, non_blocking=True)

    return json_map_leaves(_h2d, tensors)
"""
FUNCTION_REPLACEMENT = """def _nested_tensors_h2d(
    tensors: NestedTensors,
    device: torch.types.Device,
    *,
    pin_memory: bool = False,
) -> NestedTensors:
    if device is None:
        return tensors

    def _h2d(x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            return x
        if pin_memory and x.is_cpu and not (x.is_pinned() and _is_dense(x)):
            # Keep host pinning outside the measured CUDA copy interval.
            x = x.new_empty(x.shape, pin_memory=True).copy_(x)
        mm_hashes = current_mm_keys()
        request_ids = mm_request_ids_for_hashes(mm_hashes)
        should_probe_h2d = (
            os.environ.get("VL_PILOT_PROBE") == "1"
            and x.is_cpu
            and torch.device(device).type == "cuda"
            and bool(mm_hashes)
            and bool(request_ids)
        )
        if not should_probe_h2d:
            return x.to(device=device, non_blocking=True)

        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        event_start.record()
        output = x.to(device=device, non_blocking=True)
        event_end.record()
        record_stage_cuda_event(
            event_start,
            event_end,
            {
                "stage": "mm_input_h2d",
                "measurement": "single tensor nonblocking CPU-to-CUDA copy stream interval",
                "mm_hashes": mm_hashes,
                "request_ids": request_ids,
                "cuda_payload_bytes": int(x.numel() * x.element_size()),
                "tensor_shape": [int(size) for size in x.shape],
                "tensor_dtype": str(x.dtype),
                "source_is_pinned": bool(x.is_pinned()),
            },
        )
        return output

    return json_map_leaves(_h2d, tensors)
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--diff", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    source_bytes = args.snapshot.read_bytes() if args.snapshot.exists() else args.source.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise SystemExit(f"unexpected fixed E multimodal inputs source SHA-256: {source_sha}")
    source = source_bytes.decode("utf-8")
    for label, anchor in (
        ("NumPy import", IMPORT_ANCHOR),
        ("JSON tree import", HOOK_IMPORT_ANCHOR),
        ("H2D function", FUNCTION_ANCHOR),
    ):
        if source.count(anchor) != 1:
            raise SystemExit(f"expected exactly one {label} anchor, found {source.count(anchor)}")

    patched = source.replace(IMPORT_ANCHOR, IMPORT_REPLACEMENT, 1)
    patched = patched.replace(HOOK_IMPORT_ANCHOR, HOOK_IMPORT_REPLACEMENT, 1)
    patched = patched.replace(FUNCTION_ANCHOR, FUNCTION_REPLACEMENT, 1)
    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not args.snapshot.exists():
        args.snapshot.write_bytes(source_bytes)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.diff.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(patched, encoding="utf-8")
    args.diff.write_text(
        "".join(difflib.unified_diff(
            source.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile="fixed vLLM multimodal/inputs.py",
            tofile="E multimodal/inputs.py",
        )),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "component": "vllm.multimodal.inputs._nested_tensors_h2d",
        "vllm_version": "0.33.0",
        "source_sha256": source_sha,
        "patched_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "change": "records one CUDA event pair immediately around each CPU tensor's nonblocking .to(device) call; host pinning is before the start event",
        "interpretation": "CUDA stream copy interval, not end-to-end CPU preparation or a standalone PCIe bandwidth measure",
        "request_mapping": "exact hashes from the group batch context mapped through V2 scheduler request IDs",
    }
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, args.manifest)
    print(json.dumps({"status": "patched", **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
