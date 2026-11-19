#!/usr/bin/env python3
"""Build the E-only text-completion engine/API trace overlay."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path

EXPECTED_SOURCE_SHA256 = "5436049c251a416a88017cf5f84b473dad212adb58ed31acdc726b0269ee3180"
IMPORT_ANCHOR = "from vllm.vl_pilot_hook import record_server_token_event\n"
IMPORT_REPLACEMENT = (
    "from vllm.vl_pilot_hook import (\n"
    "    record_engine_result_ready,\n"
    "    record_server_token_event,\n"
    ")\n"
)
ENGINE_ANCHOR = (
    "            async for prompt_idx, res in result_generator:\n"
    "                last_res = res\n"
)
ENGINE_REPLACEMENT = (
    "            async for prompt_idx, res in result_generator:\n"
    "                # E trace: record the completion engine result before parsing.\n"
    "                record_engine_result_ready(request_id, prompt_idx, res)\n"
    "                last_res = res\n"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--diff", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise SystemExit(f"unexpected fixed vLLM completion source SHA-256: {source_sha}")
    source = source_bytes.decode("utf-8")
    if source.count(IMPORT_ANCHOR) != 1 or source.count(ENGINE_ANCHOR) != 1:
        raise SystemExit("unexpected completion serving anchors; refusing patch")
    patched = source.replace(IMPORT_ANCHOR, IMPORT_REPLACEMENT, 1)
    patched = patched.replace(ENGINE_ANCHOR, ENGINE_REPLACEMENT, 1)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.diff.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(patched, encoding="utf-8")
    args.diff.write_text(
        "".join(
            difflib.unified_diff(
                source.splitlines(keepends=True),
                patched.splitlines(keepends=True),
                fromfile="vLLM 0.33.0 completion serving.py",
                tofile="E overlay completion serving.py",
            )
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "component": "vllm.entrypoints.openai.completion.serving",
        "vllm_version": "0.33.0",
        "source_sha256": source_sha,
        "patched_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "hooks": {
            "engine_result_ready": "completion result consumed before parsing",
            "api_token_ready": "existing fixed-runtime server token hook",
            "request_id": "OpenAI completion request_id passed unchanged",
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "patched", **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
