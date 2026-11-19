#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 GPU_UUID CHUNK_SIZE PORT LOG_FILE" >&2
    exit 2
fi

gpu_uuid=$1
chunk_size=$2
port=$3
log_file=$4
experiment_root=/home/chx/project/chunk-pilot-20261106-094412
model_path="$experiment_root/model"

export CUDA_VISIBLE_DEVICES="$gpu_uuid"
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_PATH="$CUDA_HOME"
export PATH="/home/chx/projects/sglang/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec /home/chx/projects/sglang/.venv/bin/python -m sglang.launch_server \
    --model-path "$model_path" \
    --revision 1cfa9a7208912126459214e8b04321603b3df60c \
    --dtype bfloat16 \
    --kv-cache-dtype bfloat16 \
    --tp-size 1 \
    --context-length 8192 \
    --max-running-requests 8 \
    --max-total-tokens 32768 \
    --max-prefill-tokens 8192 \
    --prefill-max-requests 8 \
    --mem-fraction-static 0.75 \
    --chunked-prefill-size "$chunk_size" \
    --schedule-policy fcfs \
    --disable-radix-cache \
    --attention-backend triton \
    --decode-attention-backend triton \
    --cuda-graph-backend-decode disabled \
    --cuda-graph-backend-prefill disabled \
    --host 127.0.0.1 \
    --port "$port" \
    >"$log_file" 2>&1
