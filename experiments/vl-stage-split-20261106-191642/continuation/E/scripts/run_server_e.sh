#!/usr/bin/env bash
set -euo pipefail

RUN_ID=vl-stage-split-20261106-191642
E_P=/home/chx/project/$RUN_ID/continuation/E
OLD=/home/chx/project/vl-encode-pilot-20261106-085428
SSD_OLD=/media/chx/Win10/vl-encode-pilot-20261106-085428
E_S=/media/chx/Win10/$RUN_ID/continuation/E
GPU0_UUID=GPU-7d51d535-c0d2-577a-2a08-d6fbdd9b93a8
PORT=8285
LABEL=${VL_PILOT_LABEL:-e-trace-smoke1}
STAGE_SPLIT_MODE=${VL_STAGE_SPLIT_MODE:-N}

case "$STAGE_SPLIT_MODE" in
  N|S) ;;
  *) echo "E supports scheduler mode N or S, got: $STAGE_SPLIT_MODE" >&2; exit 2 ;;
esac

mkdir -p "$E_S/logs" "$E_S/hf-cache" "$E_S/vllm-cache" "$E_S/torch-cache" "$E_S/flashinfer" "$E_S/configs" "$E_S/features" "$E_S/torch-traces" "$E_S/patches" "$E_S/source_snapshot" "$E_S/manifests"
printf '%s\n' "{\"mode\":\"NORMAL\",\"arm\":\"E-smoke\",\"episode_id\":\"$LABEL\",\"verify\":false,\"diagnostic\":true,\"torch_profile\":false}" > "$E_S/configs/pilot_control.json"
export VL_PILOT_CONTROL_FILE="$E_S/configs/pilot_control.json"
export VL_PILOT_FEATURE_DIR="$E_S/features"
export VL_PILOT_TRACE_DIR="$E_S/torch-traces"
export VL_PILOT_PROBE="${VL_PILOT_PROBE:-1}"
export VL_PILOT_QUEUE_PROBE="${VL_PILOT_QUEUE_PROBE:-1}"
export VL_PILOT_ASYNC_LOG=1
export VL_PILOT_ASYNC_QUEUE=65536
export VL_PILOT_EVENT_FILE="$E_S/logs/$LABEL-server-events.jsonl"
export VL_PILOT_SCHED_FILE="$E_S/logs/$LABEL-scheduler-events.jsonl"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU0_UUID"
export HF_HOME="$E_S/hf-cache"
export HF_HUB_CACHE="$E_S/hf-cache/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT="$E_S/vllm-cache"
export TORCH_HOME="$E_S/torch-cache"
export TRITON_CACHE_DIR="$E_S/torch-cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$E_S/torch-cache/torchinductor"
export FLASHINFER_WORKSPACE_BASE="$E_S/flashinfer"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_SERVER_DEV_MODE=1
export PYTHONPATH="$E_S/overlay"
# Enable the E image-pruning/M-RoPE bridge with full-retention default.
# Requests can override the image keep ratio through mm_processor_kwargs.
export VL_IMAGE_KEEP_RATIO="${VL_IMAGE_KEEP_RATIO:-1.0}"
unset CUDA_HOME CUDA_PATH VL_MAX_ENCODER_INPUT_TOKENS VL_STAGE_SPLIT_PROFILE VL_STAGE_SPLIT_DECISIONS
if [[ "$STAGE_SPLIT_MODE" == S ]]; then
  export VL_STAGE_SPLIT_MODE=S
else
  unset VL_STAGE_SPLIT_MODE
fi
export CPATH="$OLD/python-runtime/cpython-3.11.13-linux-x86_64-gnu/include/python3.11"
export C_INCLUDE_PATH="$OLD/python-runtime/cpython-3.11.13-linux-x86_64-gnu/include/python3.11"

"$OLD/env/serve311/bin/python" "$E_P/scripts/patch_chat_completion_e.py" \
  --source "$OLD/env/serve311/lib/python3.11/site-packages/vllm/entrypoints/openai/chat_completion/serving.py" \
  --snapshot "$E_S/source_snapshot/chat_completion_serving.py" \
  --destination "$E_S/overlay/vllm/entrypoints/openai/chat_completion/serving.py" \
  --diff "$E_S/patches/chat_completion_serving_e.diff" \
  --manifest "$E_S/manifests/chat-hook-patch.json"
"$OLD/env/serve311/bin/python" "$E_P/scripts/patch_async_llm_e.py" \
  --source "$OLD/env/serve311/lib/python3.11/site-packages/vllm/v1/engine/async_llm.py" \
  --snapshot "$E_S/source_snapshot/async_llm.py" \
  --destination "$E_S/overlay/vllm/v1/engine/async_llm.py" \
  --diff "$E_S/patches/async_llm_e.diff" \
  --manifest "$E_S/manifests/async-llm-queue-patch.json"
"$OLD/env/serve311/bin/python" "$E_P/scripts/patch_completion_e.py" \
  --source "$OLD/env/serve311/lib/python3.11/site-packages/vllm/entrypoints/openai/completion/serving.py" \
  --destination "$E_S/overlay/vllm/entrypoints/openai/completion/serving.py" \
  --diff "$E_S/patches/completion_serving_e.diff" \
  --manifest "$E_S/manifests/completion-hook-patch.json"
"$OLD/env/serve311/bin/python" "$E_P/scripts/patch_hook_e.py" \
  --source "$E_S/overlay/vllm/vl_pilot_hook.py" \
  --snapshot "$E_S/source_snapshot/vl_pilot_hook_before_e_patch.py" \
  --destination "$E_S/overlay/vllm/vl_pilot_hook.py" \
  --diff "$E_S/patches/vl_pilot_hook_e.diff" \
  --manifest "$E_S/manifests/vl-pilot-hook-patch.json"
"$OLD/env/serve311/bin/python" "$E_P/scripts/patch_multimodal_utils_e.py" \
  --source "$OLD/env/serve311/lib/python3.11/site-packages/vllm/multimodal/utils.py" \
  --snapshot "$E_S/source_snapshot/multimodal_utils_before_e_patch.py" \
  --destination "$E_S/overlay/vllm/multimodal/utils.py" \
  --diff "$E_S/patches/multimodal_utils_e.diff" \
  --manifest "$E_S/manifests/multimodal-utils-e-patch.json"
"$OLD/env/serve311/bin/python" "$E_P/scripts/patch_multimodal_inputs_e.py" \
  --source "$OLD/env/serve311/lib/python3.11/site-packages/vllm/multimodal/inputs.py" \
  --snapshot "$E_S/source_snapshot/multimodal_inputs_before_e_patch.py" \
  --destination "$E_S/overlay/vllm/multimodal/inputs.py" \
  --diff "$E_S/patches/multimodal_inputs_e.diff" \
  --manifest "$E_S/manifests/multimodal-inputs-e-patch.json"
"$OLD/env/serve311/bin/python" "$E_P/scripts/patch_encoder_runner_e.py" \
  --source "$E_S/source_snapshot/encoder_runner_before_e_patch.py" \
  --snapshot "$E_S/source_snapshot/encoder_runner_before_e_patch.py" \
  --destination "$E_S/overlay/vllm/v1/worker/gpu/mm/encoder_runner.py" \
  --diff "$E_S/patches/encoder_runner_e.diff" \
  --manifest "$E_S/manifests/encoder-runner-e-patch.json"
"$OLD/env/serve311/bin/python" "$E_P/scripts/patch_model_runner_v2_e.py" \
  --source "$E_S/source_snapshot/gpu_model_runner_v2_before_e_patch.py" \
  --snapshot "$E_S/source_snapshot/gpu_model_runner_v2_before_e_patch.py" \
  --destination "$E_S/overlay/vllm/v1/worker/gpu/model_runner.py" \
  --diff "$E_S/patches/gpu_model_runner_v2_e.diff" \
  --manifest "$E_S/manifests/gpu-model-runner-v2-patch.json"
"$OLD/env/serve311/bin/python" "$E_P/scripts/patch_scheduler_split_e.py" \
  --source "$OLD/env/serve311/lib/python3.11/site-packages/vllm/v1/core/sched/scheduler.py" \
  --snapshot "$E_S/source_snapshot/scheduler_before_e_split.py" \
  --output "$E_S/overlay/vllm/v1/core/sched/scheduler.py" \
  --diff "$E_S/patches/scheduler_split_e.diff" \
  --manifest "$E_S/manifests/scheduler-split-e.json"
"$OLD/env/serve311/bin/python" "$E_P/scripts/patch_qwen3vl_image_pruning_e.py" \
  --source "$OLD/env/serve311/lib/python3.11/site-packages/vllm/model_executor/models/qwen3_vl.py" \
  --snapshot "$E_S/source_snapshot/qwen3_vl_before_image_pruning_e.py" \
  --output "$E_S/overlay/vllm/model_executor/models/qwen3_vl.py" \
  --diff "$E_S/patches/qwen3_vl_image_pruning_e_v0.5.diff" \
  --manifest "$E_S/manifests/qwen3-vl-image-pruning-e-v0.5.json"
"$OLD/env/serve311/bin/python" "$E_P/scripts/patch_multimodal_config_image_pruning_e.py" \
  --source "$OLD/env/serve311/lib/python3.11/site-packages/vllm/config/multimodal.py" \
  --snapshot "$E_S/source_snapshot/multimodal_config_before_image_pruning_e.py" \
  --output "$E_S/overlay/vllm/config/multimodal.py" \
  --diff "$E_S/patches/multimodal_config_image_pruning_e.diff" \
  --manifest "$E_S/manifests/multimodal-config-image-pruning-e.json"

exec "$OLD/env/serve311/bin/vllm" serve "$SSD_OLD/models/Qwen3-VL-2B-Instruct" \
  --served-model-name qwen3vlpilot \
  --tokenizer "$SSD_OLD/models/Qwen3-VL-2B-Instruct" \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --kv-cache-memory-bytes 939524096 \
  --attention-backend FLASH_ATTN \
  --mm-encoder-attn-backend FLASH_ATTN \
  --disable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --shutdown-timeout 15 \
  --stream-interval 1 \
  --enable-chunked-prefill \
  --trust-request-mm-kwargs \
  --host 127.0.0.1 \
  --port "$PORT" \
  2>&1 | tee "$E_S/logs/$LABEL-server.log"
