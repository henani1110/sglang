#!/usr/bin/env bash
set -euo pipefail

RUN_ID=vl-stage-split-20261106-191642
P=/home/chx/project/$RUN_ID
S=/media/chx/Win10/$RUN_ID
D_P=$P/continuation/D
D_S=$S/continuation/D
OLD=/home/chx/project/vl-encode-pilot-20261106-085428
SSD_OLD=/media/chx/Win10/vl-encode-pilot-20261106-085428
GPU0_UUID=GPU-7d51d535-c0d2-577a-2a08-d6fbdd9b93a8
STRATEGY=${1:?strategy required: N, F, S, A, or U}
LABEL=${2:?label required}
PROBE=${3:-1}
PORT=${4:-8271}

case "$STRATEGY" in N|F|S|A|U) ;; *) echo "unknown strategy: $STRATEGY" >&2; exit 2 ;; esac
[[ "$PROBE" == 0 || "$PROBE" == 1 ]]
[[ "$PORT" =~ ^[0-9]+$ ]]

PREFILL_INTERVAL=1
LM_BUDGET=2048
ENCODER_BUDGET=0
if [[ "$STRATEGY" == F ]]; then
  PREFILL_INTERVAL=2
  LM_BUDGET=1024
fi

mkdir -p "$D_S/logs" "$D_S/features" "$D_S/torch-traces" "$D_S/hf-cache" \
  "$D_S/vllm-cache" "$D_S/torch-cache" "$D_S/flashinfer"
printf '%s\n' '{"mode":"NORMAL","arm":"startup","episode_id":"startup","verify":false,"diagnostic":false}' > "$D_P/pilot_control.json"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU0_UUID"
unset CUDA_HOME CUDA_PATH VL_STAGE_SPLIT_MODE VL_MAX_ENCODER_INPUT_TOKENS \
  VL_STAGE_SPLIT_PROFILE VL_STAGE_SPLIT_DECISIONS
export PATH="$OLD/env/serve311/bin:$PATH"
export HF_HOME="$D_S/hf-cache"
export HF_HUB_CACHE="$D_S/hf-cache/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT="$D_S/vllm-cache"
export TORCH_HOME="$D_S/torch-cache"
export TRITON_CACHE_DIR="$D_S/torch-cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$D_S/torch-cache/torchinductor"
export FLASHINFER_WORKSPACE_BASE="$D_S/flashinfer"
export CPATH="$OLD/python-runtime/cpython-3.11.13-linux-x86_64-gnu/include/python3.11${CPATH:+:$CPATH}"
export C_INCLUDE_PATH="$OLD/python-runtime/cpython-3.11.13-linux-x86_64-gnu/include/python3.11${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export VLLM_SERVER_DEV_MODE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH="$D_P/overlay${PYTHONPATH:+:$PYTHONPATH}"
export VL_PILOT_CONTROL_FILE="$D_P/pilot_control.json"
export VL_PILOT_FEATURE_DIR="$D_S/features"
export VL_PILOT_TRACE_DIR="$D_S/torch-traces"

if [[ "$STRATEGY" == S || "$STRATEGY" == A ]]; then
  export VL_STAGE_SPLIT_MODE="$STRATEGY"
fi
if [[ "$STRATEGY" == A ]]; then
  export VL_STAGE_SPLIT_PROFILE="$D_S/manifests/estimator-profile.json"
  export VL_STAGE_SPLIT_DECISIONS="$D_S/logs/${LABEL}-decisions.jsonl"
fi
if [[ "$STRATEGY" == U ]]; then
  ENCODER_BUDGET=576
  export VL_MAX_ENCODER_INPUT_TOKENS="$ENCODER_BUDGET"
fi
if [[ "$PROBE" == 1 ]]; then
  export VL_PILOT_PROBE=1
  export VL_PILOT_ASYNC_LOG=1
  export VL_PILOT_ASYNC_QUEUE=65536
  export VL_PILOT_EVENT_FILE="${VL_PILOT_EVENT_FILE:-$D_S/logs/${LABEL}-server-events.jsonl}"
  export VL_PILOT_SCHED_FILE="${VL_PILOT_SCHED_FILE:-$D_S/logs/${LABEL}-scheduler-events.jsonl}"
else
  unset VL_PILOT_PROBE VL_PILOT_ASYNC_LOG VL_PILOT_EVENT_FILE VL_PILOT_SCHED_FILE
fi

# Chat Completions are used for image requests. Keep their serving hooks in the
# D-only overlay and verify the exact fixed vLLM source before every launch.
"$OLD/env/serve311/bin/python" "$D_P/scripts/patch_chat_completion_d.py" \
  --source "$OLD/env/serve311/lib/python3.11/site-packages/vllm/entrypoints/openai/chat_completion/serving.py" \
  --snapshot "$D_P/source_snapshot/chat_completion_serving.py" \
  --destination "$D_P/overlay/vllm/entrypoints/openai/chat_completion/serving.py" \
  --diff "$D_P/patches/chat_completion_serving_d.diff" \
  --manifest "$D_P/manifests/chat-hook-patch.json"

exec "$OLD/env/serve311/bin/vllm" serve "$SSD_OLD/models/Qwen3-VL-2B-Instruct" \
  --served-model-name qwen3vlpilot \
  --tokenizer "$SSD_OLD/models/Qwen3-VL-2B-Instruct" \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --max-num-seqs 8 \
  --max-num-batched-tokens "$LM_BUDGET" \
  --kv-cache-memory-bytes 939524096 \
  --attention-backend FLASH_ATTN \
  --mm-encoder-attn-backend FLASH_ATTN \
  --disable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --stream-interval 1 \
  --enable-chunked-prefill \
  --prefill-schedule-interval "$PREFILL_INTERVAL" \
  --host 127.0.0.1 \
  --port "$PORT"
