#!/usr/bin/env python3
"""Build the isolated D overlay for chat engine/API result tracing."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path


EXPECTED_SOURCE_SHA256 = "fbe6d3360b3a407ff4b3e3539e73d4565dc0370112a382689bc268520b86c245"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="fixed vLLM 0.33.0 source file")
    parser.add_argument("--snapshot", type=Path, required=True, help="D source snapshot destination")
    parser.add_argument("--destination", type=Path, required=True, help="D overlay source destination")
    parser.add_argument("--diff", type=Path, required=True, help="D patch diff destination")
    parser.add_argument("--manifest", type=Path, required=True, help="D patch manifest destination")
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    source_hash = sha256(source_bytes)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise SystemExit(
            f"unexpected vLLM chat serving source SHA-256 {source_hash}; "
            f"expected {EXPECTED_SOURCE_SHA256}; refuse to patch an unreviewed base"
        )
    source_text = source_bytes.decode("utf-8")

    import_anchor = "from vllm.outputs import RequestOutput\n"
    import_replacement = (
        import_anchor
        + "from vllm.vl_pilot_hook import (\n"
        + "    record_engine_result_ready,\n"
        + "    record_server_token_event,\n"
        + ")\n"
    )
    engine_anchor = "            async for res in result_generator:\n                last_res = res\n"
    engine_replacement = (
        "            async for res in result_generator:\n"
        "                # D trace: serving consumed the engine result, before chat parsing.\n"
        "                record_engine_result_ready(request_id, 0, res)\n"
        "                last_res = res\n"
    )
    api_anchor = (
        "                    yield chunk\n\n"
        "            if self._include_reasoning_tokens_details and not include_continuous_usage:\n"
    )
    api_replacement = (
        "                    # D trace: chat chunk is ready to leave the serving API.\n"
        "                    record_server_token_event(\n"
        "                        request_id,\n"
        "                        output.token_ids,\n"
        "                        res.prompt_token_ids,\n"
        "                        finish_reason is not None,\n"
        "                    )\n"
        "                    yield chunk\n\n"
        "            if self._include_reasoning_tokens_details and not include_continuous_usage:\n"
    )
    for label, anchor in (
        ("hook import", import_anchor),
        ("engine result hook", engine_anchor),
        ("chat API yield hook", api_anchor),
    ):
        count = source_text.count(anchor)
        if count != 1:
            raise SystemExit(f"expected exactly one {label} anchor, found {count}")
    patched = source_text.replace(import_anchor, import_replacement, 1)
    patched = patched.replace(engine_anchor, engine_replacement, 1)
    patched = patched.replace(api_anchor, api_replacement, 1)
    patched_bytes = patched.encode("utf-8")

    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot.exists() and args.snapshot.read_bytes() != source_bytes:
        raise SystemExit(f"existing D source snapshot differs from fixed base: {args.snapshot}")
    if not args.snapshot.exists():
        args.snapshot.write_bytes(source_bytes)

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_bytes(patched_bytes)
    args.diff.parent.mkdir(parents=True, exist_ok=True)
    diff_text = "".join(
        difflib.unified_diff(
            source_text.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile="vllm/entrypoints/openai/chat_completion/serving.py (vLLM 0.33.0)",
            tofile="D overlay chat serving.py",
        )
    )
    args.diff.write_text(diff_text, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "component": "vllm.entrypoints.openai.chat_completion.serving",
        "vllm_version": "0.33.0",
        "source_sha256": source_hash,
        "patched_sha256": sha256(patched_bytes),
        "source_snapshot": str(args.snapshot),
        "overlay_path": str(args.destination),
        "diff_path": str(args.diff),
        "hooks": {
            "engine_result_ready": "chat stream generator consumes RequestOutput, before parser/serialization",
            "api_yield": "immediately before yielding the chat chunk to SSE serialization",
            "request_id": "OpenAI chat request_id (chatcmpl-...), shared with scheduler and client response ID",
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temp = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, args.manifest)
    print(json.dumps({"status": "patched", "source_sha256": source_hash, "patched_sha256": manifest["patched_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
