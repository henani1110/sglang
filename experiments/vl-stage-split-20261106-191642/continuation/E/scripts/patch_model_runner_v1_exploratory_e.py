#!/usr/bin/env python3
"""Historical E probe for the V1 runner; the serving process actually uses V2."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path

EXPECTED_SOURCE_SHA256 = "9af35d5741574a2cbab7784e6222c6722f053f2c3de9f3f8937dbbac3e541da5"
IMPORT_ANCHOR = "import itertools\n"
IMPORT_REPLACEMENT = "import itertools\nimport os\n"
HOOK_IMPORT_ANCHOR = (
    "from vllm.vl_pilot_hook import (\n"
    "    flush_cuda_events,\n"
    "    flush_scheduler_events,\n"
    "    mm_keys_context,\n"
    "    record_scheduler_event,\n"
    ")\n"
)
HOOK_IMPORT_REPLACEMENT = (
    "from vllm.vl_pilot_hook import (\n"
    "    flush_cuda_events,\n"
    "    flush_scheduler_events,\n"
    "    mm_keys_context,\n"
    "    record_scheduler_event,\n"
    "    record_stage_diagnostic,\n"
    "    record_stage_cuda_event,\n"
    ")\n"
)
SCHEDULER_ANCHOR = "        record_scheduler_event(scheduler_output)\n"
SCHEDULER_REPLACEMENT = (
    SCHEDULER_ANCHOR
    + "        self._e_current_scheduler_output = scheduler_output\n"
)
MODEL_FORWARD_ANCHOR = """        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )
"""
MODEL_FORWARD_REPLACEMENT = """        scheduler_output = getattr(self, "_e_current_scheduler_output", None)
        probe_enabled = os.environ.get("VL_PILOT_PROBE") == "1"
        stream_capturing = (
            torch.cuda.is_current_stream_capturing() if probe_enabled else False
        )
        if probe_enabled and scheduler_output is not None:
            record_stage_diagnostic(
                {
                    "stage": "lm_model_forward_batch",
                    "stream_capturing": bool(stream_capturing),
                    "total_num_scheduled_tokens": int(
                        scheduler_output.total_num_scheduled_tokens
                    ),
                    "num_scheduled_tokens": {
                        str(key): int(value)
                        for key, value in scheduler_output.num_scheduled_tokens.items()
                    },
                    "scheduled_encoder_inputs": {
                        str(key): [int(index) for index in value]
                        for key, value in scheduler_output.scheduled_encoder_inputs.items()
                    },
                }
            )
        record_cuda_event = (
            probe_enabled
            and scheduler_output is not None
            and scheduler_output.total_num_scheduled_tokens > 0
            and not stream_capturing
        )
        event_start = event_end = None
        if record_cuda_event:
            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
            event_start.record()
        try:
            output = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )
        finally:
            if event_end is not None and event_start is not None:
                event_end.record()
                metadata = {
                    "stage": "lm_model_forward_batch",
                    "total_num_scheduled_tokens": int(
                        scheduler_output.total_num_scheduled_tokens
                    ),
                    "num_scheduled_tokens": {
                        str(key): int(value)
                        for key, value in scheduler_output.num_scheduled_tokens.items()
                    },
                    "scheduled_encoder_inputs": {
                        str(key): [int(index) for index in value]
                        for key, value in scheduler_output.scheduled_encoder_inputs.items()
                    },
                    "input_ids_shape": list(input_ids.shape) if input_ids is not None else None,
                    "inputs_embeds_shape": (
                        list(inputs_embeds.shape) if inputs_embeds is not None else None
                    ),
                }
                record_stage_cuda_event(event_start, event_end, metadata)
        return output
"""


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
        raise SystemExit(f"unexpected fixed V2 GPU runner SHA-256: {source_sha}")
    source = source_bytes.decode("utf-8")
    anchors = (
        ("import", IMPORT_ANCHOR),
        ("hook import", HOOK_IMPORT_ANCHOR),
        ("scheduler", SCHEDULER_ANCHOR),
        ("model forward", MODEL_FORWARD_ANCHOR),
    )
    for label, anchor in anchors:
        if source.count(anchor) != 1:
            raise SystemExit(f"expected exactly one {label} anchor, found {source.count(anchor)}")
    patched = source.replace(IMPORT_ANCHOR, IMPORT_REPLACEMENT, 1)
    patched = patched.replace(HOOK_IMPORT_ANCHOR, HOOK_IMPORT_REPLACEMENT, 1)
    patched = patched.replace(SCHEDULER_ANCHOR, SCHEDULER_REPLACEMENT, 1)
    patched = patched.replace(MODEL_FORWARD_ANCHOR, MODEL_FORWARD_REPLACEMENT, 1)

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.diff.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(patched, encoding="utf-8")
    args.diff.write_text(
        "".join(
            difflib.unified_diff(
                source.splitlines(keepends=True),
                patched.splitlines(keepends=True),
                fromfile="vLLM 0.33.0 gpu_model_runner.py",
                tofile="E overlay gpu_model_runner.py",
            )
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "component": "vllm.v1.worker.gpu_model_runner.GPUModelRunner",
        "vllm_version": "0.33.0",
        "source_sha256": source_sha,
        "patched_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "measurement": "CUDA-event duration around the V2 model forward batch; records scheduler token map and encoder inputs",
        "capture_guard": "skip event creation while the current stream is being captured",
        "probe_guard": "VL_PILOT_PROBE=1 and nonempty scheduler batch",
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "patched", **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
