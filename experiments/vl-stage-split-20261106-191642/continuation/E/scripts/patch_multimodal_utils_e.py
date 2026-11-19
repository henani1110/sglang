#!/usr/bin/env python3
"""Carry exact multimodal hashes into the fixed vLLM batch H2D path."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path

EXPECTED_SOURCE_SHA256 = "24a7f1f52cd609ac55da83490dd90192b6ef0b4d58a0c51101d2c10b79eb882e"
IMPORT_ANCHOR = "from vllm.inputs import MultiModalPlaceholders\n"
IMPORT_REPLACEMENT = IMPORT_ANCHOR + "from vllm.vl_pilot_hook import mm_keys_context\n"
SIGNATURE_ANCHOR = """def group_and_batch_mm_kwargs(
    mm_kwargs: list[tuple[str, MultiModalKwargsItem]],
    *,
    device: torch.types.Device = None,
    pin_memory: bool = False,
) -> Generator[tuple[str, int, BatchedTensorInputs], None, None]:
"""
SIGNATURE_REPLACEMENT = """def group_and_batch_mm_kwargs(
    mm_kwargs: list[tuple[str, MultiModalKwargsItem]],
    *,
    device: torch.types.Device = None,
    pin_memory: bool = False,
    mm_hashes: list[str] | None = None,
) -> Generator[tuple[str, int, BatchedTensorInputs], None, None]:
"""
BODY_ANCHOR = """    for modality, group in groupby(mm_kwargs, key=lambda x: x[0]):
        items_lst = [item for _, item in group]

        for num_items, mm_kwargs_batch in group_and_batch_mm_items(
            items_lst,
            device=device,
            pin_memory=pin_memory,
        ):
            yield modality, num_items, mm_kwargs_batch
"""
BODY_REPLACEMENT = """    if mm_hashes is not None and len(mm_hashes) != len(mm_kwargs):
        raise ValueError(
            f"mm_hashes must align with mm_kwargs: {len(mm_hashes)} != {len(mm_kwargs)}"
        )

    item_offset = 0
    for modality, group in groupby(mm_kwargs, key=lambda x: x[0]):
        modality_items = list(group)
        items_lst = [item for _, item in modality_items]
        modality_hashes = (
            mm_hashes[item_offset : item_offset + len(modality_items)]
            if mm_hashes is not None
            else None
        )
        item_offset += len(modality_items)

        for num_items, mm_kwargs_batch in group_and_batch_mm_items(
            items_lst,
            device=device,
            pin_memory=pin_memory,
            mm_hashes=modality_hashes,
        ):
            yield modality, num_items, mm_kwargs_batch
"""
ITEMS_SIGNATURE_ANCHOR = """def group_and_batch_mm_items(
    items: Sequence[MultiModalKwargsItem],
    *,
    device: torch.types.Device = None,
    pin_memory: bool = False,
) -> Generator[tuple[int, BatchedTensorInputs]]:
"""
ITEMS_SIGNATURE_REPLACEMENT = """def group_and_batch_mm_items(
    items: Sequence[MultiModalKwargsItem],
    *,
    device: torch.types.Device = None,
    pin_memory: bool = False,
    mm_hashes: list[str] | None = None,
) -> Generator[tuple[int, BatchedTensorInputs]]:
"""
ITEMS_BODY_ANCHOR = """    start_idx = 0
    for end_idx in range(1, len(items) + 1):
        if end_idx < len(items) and _can_batch_mm_items(
            items[end_idx - 1], items[end_idx]
        ):
            continue

        group_data = _batch_mm_items(
            items[start_idx:end_idx],
            device=device,
            pin_memory=pin_memory,
        )

        yield end_idx - start_idx, group_data
        start_idx = end_idx

    assert start_idx == len(items)
"""
ITEMS_BODY_REPLACEMENT = """    if mm_hashes is not None and len(mm_hashes) != len(items):
        raise ValueError(
            f"mm_hashes must align with items: {len(mm_hashes)} != {len(items)}"
        )

    start_idx = 0
    for end_idx in range(1, len(items) + 1):
        if end_idx < len(items) and _can_batch_mm_items(
            items[end_idx - 1], items[end_idx]
        ):
            continue

        if mm_hashes is None:
            group_data = _batch_mm_items(
                items[start_idx:end_idx],
                device=device,
                pin_memory=pin_memory,
            )
        else:
            group_hashes = mm_hashes[start_idx:end_idx]
            with mm_keys_context(group_hashes):
                group_data = _batch_mm_items(
                    items[start_idx:end_idx],
                    device=device,
                    pin_memory=pin_memory,
                )

        yield end_idx - start_idx, group_data
        start_idx = end_idx

    assert start_idx == len(items)
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
        raise SystemExit(f"unexpected fixed E multimodal utils source SHA-256: {source_sha}")
    source = source_bytes.decode("utf-8")
    anchors = (
        ("hook import", IMPORT_ANCHOR),
        ("item batch signature", ITEMS_SIGNATURE_ANCHOR),
        ("item batch body", ITEMS_BODY_ANCHOR),
        ("batch function signature", SIGNATURE_ANCHOR),
        ("batch grouping body", BODY_ANCHOR),
    )
    for label, anchor in anchors:
        if source.count(anchor) != 1:
            raise SystemExit(f"expected exactly one {label} anchor, found {source.count(anchor)}")

    patched = source.replace(IMPORT_ANCHOR, IMPORT_REPLACEMENT, 1)
    patched = patched.replace(ITEMS_SIGNATURE_ANCHOR, ITEMS_SIGNATURE_REPLACEMENT, 1)
    patched = patched.replace(ITEMS_BODY_ANCHOR, ITEMS_BODY_REPLACEMENT, 1)
    patched = patched.replace(SIGNATURE_ANCHOR, SIGNATURE_REPLACEMENT, 1)
    patched = patched.replace(BODY_ANCHOR, BODY_REPLACEMENT, 1)
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
            fromfile="fixed vLLM multimodal/utils.py",
            tofile="E multimodal/utils.py",
        )),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "component": "vllm.multimodal.utils.group_and_batch_mm_items/group_and_batch_mm_kwargs",
        "vllm_version": "0.33.0",
        "source_sha256": source_sha,
        "patched_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "change": "optional aligned mm_hashes context is entered before each exact item group is tensor-reduced and moved to the device; batching behavior unchanged when mm_hashes is omitted",
        "measurement_role": "propagate request/image identity into the actual nested H2D copy path",
    }
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, args.manifest)
    print(json.dumps({"status": "patched", **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
