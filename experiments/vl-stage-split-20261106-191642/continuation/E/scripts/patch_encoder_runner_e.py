#!/usr/bin/env python3
"""Instrument E-owned multimodal batch preparation and H2D copy events."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path

EXPECTED_SOURCE_SHA256 = "1f8a51ed3b348dcdc086279a2ac8edc23e417464e760af51baf759b721f6c5eb"
IMPORT_OS_ANCHOR = "import time\n"
IMPORT_OS_REPLACEMENT = "import time\nimport os\n"
HOOK_IMPORT_ANCHOR = "from vllm.vl_pilot_hook import mm_keys_context\n"
HOOK_IMPORT_REPLACEMENT = """from vllm.vl_pilot_hook import (
    mm_keys_context,
    mm_request_ids_for_hashes,
    record_stage_diagnostic,
)
"""
LOGGER_ANCHOR = "logger = init_logger(__name__)\n"
HELPER = """
def _pilot_cuda_tensor_bytes(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size()) if value.is_cuda else 0
    if isinstance(value, dict):
        return sum(_pilot_cuda_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_pilot_cuda_tensor_bytes(item) for item in value)
    return 0


"""
LOOP_ANCHOR = """        for modality, num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(
            [mm_kwargs[i] for i in order], device=self.device, pin_memory=PIN_MEMORY
        ):
            batch_keys = (
                [mm_hashes[i] for i in order[current_item_idx : current_item_idx + num_items]]
                if mm_hashes is not None
                else []
            )
"""
LOOP_REPLACEMENT = """        batch_iterator = iter(
            group_and_batch_mm_kwargs(
                [mm_kwargs[i] for i in order],
                device=self.device,
                pin_memory=PIN_MEMORY,
                mm_hashes=(
                    [mm_hashes[i] for i in order]
                    if mm_hashes is not None
                    else None
                ),
            )
        )
        while True:
            probe_batch = (
                os.environ.get("VL_PILOT_PROBE") == "1"
                and mm_hashes is not None
                and getattr(self.device, "type", None) == "cuda"
            )
            host_batch_start_ns = time.perf_counter_ns()
            try:
                modality, num_items, mm_kwargs_batch = next(batch_iterator)
            except StopIteration:
                break
            batch_keys = (
                [mm_hashes[i] for i in order[current_item_idx : current_item_idx + num_items]]
                if mm_hashes is not None
                else []
            )
            if probe_batch:
                record_stage_diagnostic(
                    {
                        "stage": "mm_batch_prepare_enqueue",
                        "modality": str(modality),
                        "num_items": int(num_items),
                        "mm_hashes": [str(key) for key in batch_keys],
                        "request_ids": mm_request_ids_for_hashes(batch_keys),
                        "host_batch_prepare_enqueue_wall_ms": (
                            time.perf_counter_ns() - host_batch_start_ns
                        ) / 1e6,
                        "cuda_payload_bytes": _pilot_cuda_tensor_bytes(mm_kwargs_batch),
                        "interpretation": (
                            "host wall envelope includes Python batching and async H2D enqueue; "
                            "CUDA copy interval is recorded separately at tensor .to"
                        ),
                    }
                )
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
        raise SystemExit(f"unexpected fixed E encoder_runner source SHA-256: {source_sha}")
    source = source_bytes.decode("utf-8")
    anchors = (
        ("time import", IMPORT_OS_ANCHOR),
        ("pilot hook import", HOOK_IMPORT_ANCHOR),
        ("logger insertion", LOGGER_ANCHOR),
        ("encoder batch loop", LOOP_ANCHOR),
    )
    for label, anchor in anchors:
        if source.count(anchor) != 1:
            raise SystemExit(f"expected exactly one {label} anchor, found {source.count(anchor)}")

    patched = source.replace(IMPORT_OS_ANCHOR, IMPORT_OS_REPLACEMENT, 1)
    patched = patched.replace(HOOK_IMPORT_ANCHOR, HOOK_IMPORT_REPLACEMENT, 1)
    patched = patched.replace(LOGGER_ANCHOR, HELPER + LOGGER_ANCHOR, 1)
    patched = patched.replace(LOOP_ANCHOR, LOOP_REPLACEMENT, 1)

    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not args.snapshot.exists():
        args.snapshot.write_bytes(source_bytes)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.diff.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(patched, encoding="utf-8")
    args.diff.write_text(
        "".join(
            difflib.unified_diff(
                source.splitlines(keepends=True),
                patched.splitlines(keepends=True),
                fromfile="fixed vLLM encoder_runner.py",
                tofile="E encoder_runner.py",
            )
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "component": "vllm.v1.worker.gpu.mm.encoder_runner.EncoderRunner.execute_mm_encoder",
        "vllm_version": "0.33.0",
        "source_sha256": source_sha,
        "patched_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "measurement": "host batching/enqueue wall envelope in diagnostics; actual per-tensor H2D CUDA stream interval is instrumented at _nested_tensors_h2d",
        "request_mapping": "mm_hashes map through V2 request-id context",
        "probe_guard": "VL_PILOT_PROBE=1 and request mm_hashes are present; excludes startup encoder profiling",
    }
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, args.manifest)
    print(json.dumps({"status": "patched", **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
