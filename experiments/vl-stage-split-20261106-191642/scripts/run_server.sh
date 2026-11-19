#!/usr/bin/env bash
set -euo pipefail

RUN_ID=vl-stage-split-20261106-191642
P=/home/chx/project/$RUN_ID
S=/media/chx/Win10/$RUN_ID
OLD=/home/chx/project/vl-encode-pilot-20261106-085428
SSD_OLD=/media/chx/Win10/vl-encode-pilot-20261106-085428
GPU0_UUID=GPU-7d51d535-c0d2-577a-2a08-d6fbdd9b93a8
CONFIG=${1:?configuration required: eager or default}
PROBE=${2:?probe flag required: 0 or 1}
LABEL=${3:-${CONFIG}-probe${PROBE}}

case "$CONFIG" in
  eager) EXTRA=(--no-async-scheduling --enforce-eager) ;;
  default) EXTRA=() ;;
  *) echo "unknown config: $CONFIG" >&2; exit 2 ;;
esac
case "$PROBE" in
  0|1) ;;
  *) echo "probe must be 0 or 1" >&2; exit 2 ;;
esac

printf '%s\n' '{"mode":"NORMAL","arm":"startup","episode_id":"startup","verify":false,"diagnostic":false}' > "$P/pilot_control.json"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU0_UUID"
unset CUDA_HOME CUDA_PATH
export PATH="$OLD/env/serve311/bin:$PATH"
export HF_HOME="$S/hf-cache"
export HF_HUB_CACHE="$S/hf-cache/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT="$S/vllm-cache"
export TORCH_HOME="$S/torch-cache"
export TRITON_CACHE_DIR="$S/torch-cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$S/torch-cache/torchinductor"
export FLASHINFER_WORKSPACE_BASE="$S"
export CPATH="$OLD/python-runtime/cpython-3.11.13-linux-x86_64-gnu/include/python3.11${CPATH:+:$CPATH}"
export C_INCLUDE_PATH="$OLD/python-runtime/cpython-3.11.13-linux-x86_64-gnu/include/python3.11${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export VLLM_SERVER_DEV_MODE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH="$P/overlay${PYTHONPATH:+:$PYTHONPATH}"
export VL_PILOT_CONTROL_FILE="$P/pilot_control.json"
export VL_PILOT_FEATURE_DIR="$S/features"
export VL_PILOT_TRACE_DIR="$S/logs/torch-traces"

if [[ "$PROBE" == 1 ]]; then
  export VL_PILOT_PROBE=1
  export VL_PILOT_EVENT_FILE="$S/logs/${LABEL}-server-events.jsonl"
  export VL_PILOT_SCHED_FILE="$S/logs/${LABEL}-scheduler-events.jsonl"
else
  unset VL_PILOT_PROBE VL_PILOT_EVENT_FILE VL_PILOT_SCHED_FILE
fi

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
  --stream-interval 1 \
  --enable-chunked-prefill \
  "${EXTRA[@]}" \
  --host 127.0.0.1 \
  --port 8271
