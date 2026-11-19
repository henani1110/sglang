#!/usr/bin/env python3
"""Build an isolated vLLM AsyncLLM overlay with opt-in queue timing events."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path


EXPECTED_SOURCE_SHA256 = "73ef3bfe5352efa4104fd36a6053e5044f5347b36b197a745361dfe195d5800e"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--diff", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise SystemExit(
            f"unexpected AsyncLLM source SHA-256 {source_sha}; "
            f"expected {EXPECTED_SOURCE_SHA256}; refusing unreviewed base"
        )
    source = source_bytes.decode("utf-8")

    replacements = [
        (
            "import vllm.envs as envs\n",
            "import vllm.envs as envs\nfrom vllm.vl_pilot_hook import record_queue_event\n",
            "queue hook import",
        ),
        (
            '        """Add new request to the AsyncLLM."""\n        if self.errored:\n',
            '        """Add new request to the AsyncLLM."""\n'
            '        record_queue_event("engine_input_start", request_id)\n'
            '        if self.errored:\n',
            "input processing start",
        ),
        (
            "        self.input_processor.assign_request_id(request)\n\n"
            "        # We start the output_handler on the first call to add_request()",
            "        self.input_processor.assign_request_id(request)\n"
            '        record_queue_event(\n'
            '            "engine_input_ready", request_id, engine_request_id=request.request_id\n'
            "        )\n\n"
            "        # We start the output_handler on the first call to add_request()",
            "input processing ready",
        ),
        (
            "            await self._add_request(request, prompt_text, None, 0, queue)\n",
            "            await self._add_request(\n"
            "                request, prompt_text, None, 0, queue, public_request_id=request_id\n"
            "            )\n",
            "public request ID forwarding",
        ),
        (
            "        index: int,\n        queue: RequestOutputCollector,\n    ):\n",
            "        index: int,\n"
            "        queue: RequestOutputCollector,\n"
            "        public_request_id: str | None = None,\n"
            "    ):\n",
            "submit helper signature",
        ),
        (
            "        # Add the EngineCoreRequest to EngineCore (separate process).\n"
            "        await self.engine_core.add_request_async(request)\n",
            "        # Add the EngineCoreRequest to EngineCore (separate process).\n"
            '        record_queue_event(\n'
            '            "engine_submit",\n'
            '            public_request_id or request.request_id,\n'
            '            engine_request_id=request.request_id,\n'
            "        )\n"
            "        await self.engine_core.add_request_async(request)\n",
            "engine submit hook",
        ),
        (
            "                await self.engine_core.add_request_async(child_request)\n",
            "                record_queue_event(\n"
            '                    "engine_submit", request_id, engine_request_id=child_request.request_id\n'
            "                )\n"
            "                await self.engine_core.add_request_async(child_request)\n",
            "child engine submit hook",
        ),
        (
            "                    await self._add_request(req, prompt_text, None, 0, queue)\n",
            "                    await self._add_request(\n"
            "                        req, prompt_text, None, 0, queue, public_request_id=request_id\n"
            "                    )\n",
            "streaming input request ID forwarding",
        ),
        (
            "                            await self._add_request(final_req, None, None, 0, queue)\n",
            "                            await self._add_request(\n"
            "                                final_req, None, None, 0, queue, public_request_id=request_id\n"
            "                            )\n",
            "stream final request ID forwarding",
        ),
    ]
    patched = source
    for anchor, replacement, label in replacements:
        count = patched.count(anchor)
        if count != 1:
            raise SystemExit(f"expected exactly one {label} anchor, found {count}")
        patched = patched.replace(anchor, replacement, 1)

    patched_bytes = patched.encode("utf-8")
    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot.exists() and args.snapshot.read_bytes() != source_bytes:
        raise SystemExit(f"existing snapshot differs from pinned source: {args.snapshot}")
    if not args.snapshot.exists():
        args.snapshot.write_bytes(source_bytes)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_bytes(patched_bytes)
    args.diff.parent.mkdir(parents=True, exist_ok=True)
    args.diff.write_text(
        "".join(
            difflib.unified_diff(
                source.splitlines(keepends=True),
                patched.splitlines(keepends=True),
                fromfile="vllm/v1/engine/async_llm.py (pinned vLLM 0.33.0)",
                tofile="E overlay AsyncLLM queue probe",
            )
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "component": "vllm.v1.engine.async_llm.AsyncLLM",
        "vllm_version": "0.33.0",
        "source_sha256": source_sha,
        "patched_sha256": hashlib.sha256(patched_bytes).hexdigest(),
        "hooks": {
            "engine_input_start": "AsyncLLM.add_request entry before input processing",
            "engine_input_ready": "after input processor assigns the engine request ID",
            "engine_submit": "immediately before EngineCore.add_request_async",
        },
        "probe_gate": "VL_PILOT_QUEUE_PROBE=1",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temp = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, args.manifest)
    print(json.dumps({"status": "patched", **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
