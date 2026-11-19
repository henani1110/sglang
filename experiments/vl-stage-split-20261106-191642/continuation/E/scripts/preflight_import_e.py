#!/usr/bin/env python3
"""Verify the isolated E runtime overlay resolves to E-owned modules."""
from __future__ import annotations

import inspect
import json

import vllm
import vllm.entrypoints.openai.chat_completion.serving as chat_serving
import vllm.v1.core.sched.scheduler as scheduler_module
import vllm.v1.worker.gpu.model_runner as gpu_model_runner_v2_module
import vllm.v1.worker.gpu_worker as gpu_worker_module
import vllm.v1.worker.gpu.mm.encoder_runner as encoder_runner_module
import vllm.multimodal.inputs as multimodal_inputs_module
import vllm.multimodal.utils as multimodal_utils_module
import vllm.vl_pilot_hook as pilot_hook


def main() -> None:
    scheduler_source = inspect.getsource(scheduler_module.Scheduler)
    encoder_runner_source = inspect.getsource(encoder_runner_module.EncoderRunner.execute_mm_encoder)
    h2d_source = inspect.getsource(multimodal_inputs_module._nested_tensors_h2d)
    batch_source = inspect.getsource(multimodal_utils_module.group_and_batch_mm_kwargs)
    runner = gpu_model_runner_v2_module.GPUModelRunner
    worker_source = inspect.getsource(gpu_worker_module.Worker._make_model_runner)
    result = {
        "vllm_version": vllm.__version__,
        "chat_serving": chat_serving.__file__,
        "scheduler": scheduler_module.__file__,
        "scheduler_contains_d_policy": "d_policy_decision" in scheduler_source,
        "gpu_model_runner_v2": gpu_model_runner_v2_module.__file__,
        "gpu_worker_selects_v2_runner": "vllm.v1.worker.gpu.model_runner" in worker_source,
        "has_pilot_flush": hasattr(runner, "pilot_flush_visual_events"),
        "has_e_stage_cuda_event": hasattr(pilot_hook, "record_stage_cuda_event"),
        "has_e_stage_diagnostic": hasattr(pilot_hook, "record_stage_diagnostic"),
        "has_mm_request_id_context": hasattr(pilot_hook, "mm_request_ids_context"),
        "has_mm_request_lookup": hasattr(pilot_hook, "mm_request_ids_for_hashes"),
        "has_current_mm_keys": hasattr(pilot_hook, "current_mm_keys"),
        "encoder_runner": encoder_runner_module.__file__,
        "has_h2d_event_probe": '"stage": "mm_input_h2d"' in h2d_source,
        "has_hash_aligned_batch_context": "mm_hashes" in batch_source,
        "has_host_batch_diagnostic": '"stage": "mm_batch_prepare_enqueue"' in encoder_runner_source,
        "multimodal_inputs": multimodal_inputs_module.__file__,
        "multimodal_utils": multimodal_utils_module.__file__,
        "pilot_hook": pilot_hook.__file__,
    }
    print(json.dumps(result, ensure_ascii=False))
    assert result["vllm_version"] == "0.33.0"
    assert "/continuation/E/overlay/" in result["chat_serving"]
    assert "/continuation/E/overlay/" in result["scheduler"]
    assert not result["scheduler_contains_d_policy"]
    assert "/continuation/E/overlay/" in result["gpu_model_runner_v2"]
    assert result["gpu_worker_selects_v2_runner"]
    assert result["has_pilot_flush"]
    assert result["has_e_stage_cuda_event"]
    assert result["has_e_stage_diagnostic"]
    assert result["has_mm_request_id_context"]
    assert result["has_mm_request_lookup"]
    assert result["has_current_mm_keys"]
    assert "/continuation/E/overlay/" in result["encoder_runner"]
    assert result["has_h2d_event_probe"]
    assert result["has_hash_aligned_batch_context"]
    assert result["has_host_batch_diagnostic"]
    assert "/continuation/E/overlay/" in result["multimodal_inputs"]
    assert "/continuation/E/overlay/" in result["multimodal_utils"]
    assert "/continuation/E/overlay/" in result["pilot_hook"]


if __name__ == "__main__":
    main()
