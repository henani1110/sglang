#!/usr/bin/env bash
set -euo pipefail

P=/home/chx/project/vl-encode-pilot-20261106-085428
S=/media/chx/Win10/vl-encode-pilot-20261106-085428
GPU1_UUID=GPU-321a3dbc-a3ed-3f07-3b13-31dd67f143b0
PYTHON_HEADERS="$P/python-runtime/cpython-3.11.13-linux-x86_64-gnu/include/python3.11"

printf '%s\n' '{"mode":"NORMAL","arm":"startup-gpu1","episode_id":"startup-gpu1","verify":false,"diagnostic":false,"torch_profile":false}' > "$P/pilot_control.json"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU1_UUID"
unset CUDA_HOME CUDA_PATH
export PATH="$P/env/serve311/bin:$PATH"
export HF_HOME="$S/hf-cache"
export HF_HUB_CACHE="$S/hf-cache/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$S/torch-cache"
export TRITON_CACHE_DIR="$S/torch-cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$S/torch-cache/torchinductor"
export FLASHINFER_WORKSPACE_BASE="$S"
export CPATH="$PYTHON_HEADERS${CPATH:+:$CPATH}"
export C_INCLUDE_PATH="$PYTHON_HEADERS${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export VLLM_SERVER_DEV_MODE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VL_PILOT_CONTROL_FILE="$P/pilot_control.json"
export VL_PILOT_FEATURE_DIR="$S/features-gpu1"
export VL_PILOT_EVENT_FILE="$S/logs/server_events_gpu1.jsonl"
export VL_PILOT_SCHED_FILE="$P/logs/scheduler_events_gpu1.jsonl"
export VL_PILOT_TRACE_DIR="$S/logs/torch-traces-gpu1"

exec "$P/env/serve311/bin/vllm" serve "$S/models/Qwen3-VL-2B-Instruct" \
  --served-model-name qwen3vlpilot \
  --tokenizer "$S/models/Qwen3-VL-2B-Instruct" \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --kv-cache-memory-bytes 939524096 \
  --attention-backend FLASH_ATTN \
  --mm-encoder-attn-backend FLASH_ATTN \
  --disable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --stream-interval 1 \
  --enable-chunked-prefill \
  --no-async-scheduling \
  --enforce-eager \
  --host 127.0.0.1 \
  --port 8262
