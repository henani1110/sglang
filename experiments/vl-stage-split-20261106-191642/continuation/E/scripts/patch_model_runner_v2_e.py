#!/usr/bin/env python3
"""Instrument the V2 GPU runner's actual graph replay/eager launch boundaries."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path

EXPECTED_SOURCE_SHA256 = "a4094c87c4f5f4df0a6d0009247e587be2019e2446439bc84f2c490911c4b898"
IMPORT_ANCHOR = "import gc\n"
IMPORT_REPLACEMENT = "import gc\nimport os\n"
HOOK_IMPORT_ANCHOR = (
    "from vllm.vl_pilot_hook import (\n"
    "    flush_cuda_events,\n"
    "    flush_scheduler_events,\n"
    "    record_scheduler_event,\n"
    ")\n"
)
HOOK_IMPORT_REPLACEMENT = (
    "from vllm.vl_pilot_hook import (\n"
    "    flush_cuda_events,\n"
    "    flush_scheduler_events,\n"
    "    mm_request_ids_context,\n"
    "    record_scheduler_event,\n"
    "    record_stage_cuda_event,\n"
    "    record_stage_diagnostic,\n"
    ")\n"
)
SCHEDULER_CALL_ANCHOR = "            record_scheduler_event(scheduler_output)\n"
SCHEDULER_CALL_REPLACEMENT = "            record_scheduler_event(scheduler_output, self.kv_cache_config)\n"
MM_EMBED_ANCHOR = '''                with self.ec_connector.maybe_get_output(
                    scheduler_output
                ) as ec_connector_output:
                    inputs_embeds = self.model_state.prepare_inputs_embeds(
                        scheduled_encoder_inputs, input_batch, self.req_states
                    )
'''
MM_EMBED_REPLACEMENT = '''                with self.ec_connector.maybe_get_output(
                    scheduler_output
                ) as ec_connector_output:
                    mm_request_ids: dict[str, set[str]] = {}
                    encoder_cache = getattr(self, "encoder_cache", None)
                    mm_features_by_req = getattr(encoder_cache, "mm_features", {})
                    for request_id, input_indices in scheduled_encoder_inputs.items():
                        features = mm_features_by_req.get(request_id, ())
                        for input_index in input_indices:
                            index = int(input_index)
                            if 0 <= index < len(features):
                                mm_hash = getattr(features[index], "identifier", None)
                                if mm_hash is not None:
                                    mm_request_ids.setdefault(str(mm_hash), set()).add(
                                        str(request_id)
                                    )
                    with mm_request_ids_context(
                        {key: sorted(value) for key, value in mm_request_ids.items()}
                    ):
                        inputs_embeds = self.model_state.prepare_inputs_embeds(
                            scheduled_encoder_inputs, input_batch, self.req_states
                        )
'''
FORWARD_HELPER_ANCHOR = (
    "        self.step_timing.forward_start()\n\n"
    "        # Run model.\n"
)
FORWARD_HELPER_REPLACEMENT = """        self.step_timing.forward_start()

        def run_model_with_probe(stage, run):
            probe_enabled = (
                os.environ.get("VL_PILOT_PROBE") == "1"
                and not dummy_run
                and not is_profile
                and not any(
                    str(request_id).startswith("_warmup_")
                    for request_id in scheduler_output.num_scheduled_tokens
                )
            )
            stream_capturing = (
                torch.cuda.is_current_stream_capturing() if probe_enabled else False
            )
            metadata = {
                "stage": stage,
                "cuda_graph_mode": getattr(batch_desc.cg_mode, "name", str(batch_desc.cg_mode)),
                "num_tokens_after_padding": int(input_batch.num_tokens_after_padding),
                "num_reqs": int(input_batch.num_reqs),
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
            if probe_enabled:
                record_stage_diagnostic(
                    {**metadata, "stream_capturing": bool(stream_capturing)}
                )
            if not probe_enabled or stream_capturing:
                return run()

            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
            event_start.record()
            host_submit_start_ns = time.perf_counter_ns()
            try:
                output = run()
            finally:
                event_end.record()
                metadata["host_submit_ms"] = (
                    time.perf_counter_ns() - host_submit_start_ns
                ) / 1e6
                record_stage_cuda_event(event_start, event_end, metadata)
            return output

        # Run model.
"""
FULLGRAPH_ANCHOR = "            model_output = self.cudagraph_manager.run_fullgraph(batch_desc)\n"
FULLGRAPH_REPLACEMENT = """            model_output = run_model_with_probe(
                "lm_fullgraph_replay",
                lambda: self.cudagraph_manager.run_fullgraph(batch_desc),
            )
"""
UBATCH_ANCHOR = """                    model_output = self.ubatch_runner.run(
                        self.model, model_inputs, ubatch_state
                    )
"""
UBATCH_REPLACEMENT = """                    model_output = run_model_with_probe(
                        "lm_ubatch_forward",
                        lambda: self.ubatch_runner.run(
                            self.model, model_inputs, ubatch_state
                        ),
                    )
"""
PIECEWISE_ANCHOR = """                    model_output = self.cudagraph_manager.run_pw_graph(
                        self.model, model_inputs
                    )
"""
PIECEWISE_REPLACEMENT = """                    model_output = run_model_with_probe(
                        "lm_piecewise_replay",
                        lambda: self.cudagraph_manager.run_pw_graph(
                            self.model, model_inputs
                        ),
                    )
"""
EAGER_ANCHOR = "                    model_output = self.model(**model_inputs)\n"
EAGER_REPLACEMENT = """                    model_output = run_model_with_probe(
                        "lm_eager_forward", lambda: self.model(**model_inputs)
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
        raise SystemExit(f"unexpected fixed V2 GPU runner SHA-256: {source_sha}")
    source = source_bytes.decode("utf-8")
    anchors = (
        ("import", IMPORT_ANCHOR),
        ("hook import", HOOK_IMPORT_ANCHOR),
        ("scheduler event call", SCHEDULER_CALL_ANCHOR),
        ("multimodal embed call", MM_EMBED_ANCHOR),
        ("forward helper", FORWARD_HELPER_ANCHOR),
        ("fullgraph replay", FULLGRAPH_ANCHOR),
        ("ubatch forward", UBATCH_ANCHOR),
        ("piecewise replay", PIECEWISE_ANCHOR),
        ("eager forward", EAGER_ANCHOR),
    )
    for label, anchor in anchors:
        count = source.count(anchor)
        if count != 1:
            raise SystemExit(f"expected exactly one {label} anchor, found {count}")

    patched = source.replace(IMPORT_ANCHOR, IMPORT_REPLACEMENT, 1)
    patched = patched.replace(HOOK_IMPORT_ANCHOR, HOOK_IMPORT_REPLACEMENT, 1)
    patched = patched.replace(SCHEDULER_CALL_ANCHOR, SCHEDULER_CALL_REPLACEMENT, 1)
    patched = patched.replace(MM_EMBED_ANCHOR, MM_EMBED_REPLACEMENT, 1)
    patched = patched.replace(FORWARD_HELPER_ANCHOR, FORWARD_HELPER_REPLACEMENT, 1)
    patched = patched.replace(FULLGRAPH_ANCHOR, FULLGRAPH_REPLACEMENT, 1)
    patched = patched.replace(UBATCH_ANCHOR, UBATCH_REPLACEMENT, 1)
    patched = patched.replace(PIECEWISE_ANCHOR, PIECEWISE_REPLACEMENT, 1)
    patched = patched.replace(EAGER_ANCHOR, EAGER_REPLACEMENT, 1)

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
                fromfile="vLLM V2 GPU model_runner.py fixed E base",
                tofile="E V2 GPU model_runner.py",
            )
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "component": "vllm.v1.worker.gpu.model_runner.GPUModelRunner",
        "vllm_version": "0.33.0",
        "source_sha256": source_sha,
        "patched_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "measurement": "CUDA events and host submit duration around actual FULL graph replay, PIECEWISE replay, ubatch or eager model launch; per-request KV block IDs and runtime page geometry",
        "trace_metadata": "scheduler request token map; visual encoder mm_hash to all scheduled request IDs; scheduled encoder input map",
        "probe_guard": "VL_PILOT_PROBE=1; excludes dummy, profile and _warmup_ runs",
        "capture_guard": "skip event recording if current stream is being captured",
    }
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, args.manifest)
    print(json.dumps({"status": "patched", **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
