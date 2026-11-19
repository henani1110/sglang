#!/usr/bin/env bash
set -euo pipefail

RUN_ID=vl-stage-split-20261106-191642
E_P=/home/chx/project/$RUN_ID/continuation/E
E_S=/media/chx/Win10/$RUN_ID/continuation/E
D_P=/home/chx/project/$RUN_ID/continuation/D
D_S=/media/chx/Win10/$RUN_ID/continuation/D
OLD=/home/chx/project/vl-encode-pilot-20261106-085428
SSD_OLD=/media/chx/Win10/vl-encode-pilot-20261106-085428
GPU0_UUID=GPU-7d51d535-c0d2-577a-2a08-d6fbdd9b93a8
PORT=8285
LABEL=${VL_PILOT_LABEL:-e-deadline-a}

mkdir -p "$E_S/logs" "$E_S/hf-cache" "$E_S/vllm-cache" "$E_S/torch-cache" \
  "$E_S/flashinfer" "$E_S/configs" "$E_S/features" "$E_S/torch-traces"
printf '%s\n' "{\"mode\":\"NORMAL\",\"arm\":\"A\",\"episode_id\":\"$LABEL\",\"verify\":false,\"diagnostic\":true,\"torch_profile\":false}" > "$E_S/configs/pilot_control.json"

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
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$D_P/overlay"
export VL_PILOT_CONTROL_FILE="$E_S/configs/pilot_control.json"
export VL_PILOT_FEATURE_DIR="$E_S/features"
export VL_PILOT_TRACE_DIR="$E_S/torch-traces"
export VL_PILOT_PROBE=1
export VL_PILOT_QUEUE_PROBE=1
export VL_PILOT_ASYNC_LOG=1
export VL_PILOT_ASYNC_QUEUE=65536
export VL_PILOT_EVENT_FILE="$E_S/logs/$LABEL-server-events.jsonl"
export VL_PILOT_SCHED_FILE="$E_S/logs/$LABEL-scheduler-events.jsonl"
export VL_STAGE_SPLIT_MODE=A
export VL_STAGE_SPLIT_PROFILE="$D_S/manifests/estimator-profile.json"
export VL_STAGE_SPLIT_DECISIONS="$E_S/logs/$LABEL-deadline-decisions.jsonl"
unset CUDA_HOME CUDA_PATH VL_MAX_ENCODER_INPUT_TOKENS VL_IMAGE_KEEP_RATIO
export CPATH="$OLD/python-runtime/cpython-3.11.13-linux-x86_64-gnu/include/python3.11"
export C_INCLUDE_PATH="$OLD/python-runtime/cpython-3.11.13-linux-x86_64-gnu/include/python3.11"

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
  --prefill-schedule-interval 1 \
  --host 127.0.0.1 \
  --port "$PORT"
