#!/usr/bin/env bash
set -euo pipefail

RUN_ID=vl-stage-split-20261106-191642
OLD=/home/chx/project/vl-encode-pilot-20261106-085428
SSD_OLD=/media/chx/Win10/vl-encode-pilot-20261106-085428
E_S=/media/chx/Win10/$RUN_ID/continuation/E
MODEL=$E_S/models/Qwen3-VL-4B-Instruct
GPU0_UUID=GPU-7d51d535-c0d2-577a-2a08-d6fbdd9b93a8
PORT=${VL_PILOT_PORT:-8286}
LABEL=${VL_PILOT_LABEL:-docvqa-qwen3vl4b-capacity-reference}
IMAGE_ROOT=$E_S/datasets/manifests/docvqa-2026-validation-v0.1/images

mkdir -p "$E_S/logs" "$E_S/hf-cache" "$E_S/vllm-cache" "$E_S/torch-cache" \
  "$E_S/flashinfer" "$E_S/configs" "$E_S/features/$LABEL" \
  "$E_S/torch-traces/$LABEL"
printf '%s\n' "{\"mode\":\"NORMAL\",\"arm\":\"DocVQA-long-context-smoke\",\"episode_id\":\"$LABEL\",\"verify\":false,\"diagnostic\":true,\"torch_profile\":false}" \
  > "$E_S/configs/$LABEL-pilot_control.json"

export VL_PILOT_CONTROL_FILE="$E_S/configs/$LABEL-pilot_control.json"
export VL_PILOT_FEATURE_DIR="$E_S/features/$LABEL"
export VL_PILOT_TRACE_DIR="$E_S/torch-traces/$LABEL"
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
export VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES="${VL_PILOT_MAX_MEDIA_REQUEST_DECODE_BYTES:-1073741824}"
export PYTHONPATH="$E_S/overlay"
export VL_IMAGE_KEEP_RATIO=1.0
unset CUDA_HOME CUDA_PATH VL_MAX_ENCODER_INPUT_TOKENS VL_STAGE_SPLIT_PROFILE VL_STAGE_SPLIT_DECISIONS VL_STAGE_SPLIT_MODE
export CPATH="$OLD/python-runtime/cpython-3.11.13-linux-x86_64-gnu/include/python3.11"
export C_INCLUDE_PATH="$OLD/python-runtime/cpython-3.11.13-linux-x86_64-gnu/include/python3.11"

exec "$OLD/env/serve311/bin/vllm" serve "$MODEL" \
  --served-model-name qwen3vl4b \
  --tokenizer "$MODEL" \
  --dtype bfloat16 \
  --max-model-len "${VL_PILOT_MAX_MODEL_LEN:-50000}" \
  --max-num-seqs "${VL_PILOT_MAX_NUM_SEQS:-1}" \
  --max-num-batched-tokens "${VL_PILOT_MAX_NUM_BATCHED_TOKENS:-4096}" \
  --kv-cache-memory-bytes 8000000000 \
  --attention-backend FLASH_ATTN \
  --mm-encoder-attn-backend FLASH_ATTN \
  --disable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --shutdown-timeout 15 \
  --stream-interval 1 \
  --enable-chunked-prefill \
  --enforce-eager \
  --trust-request-mm-kwargs \
  --limit-mm-per-prompt '{"image":181}' \
  --allowed-local-media-path "$IMAGE_ROOT" \
  --host 127.0.0.1 \
  --port "$PORT" \
  2>&1 | tee "$E_S/logs/$LABEL-server.log"
