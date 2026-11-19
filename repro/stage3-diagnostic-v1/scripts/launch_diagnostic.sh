#!/usr/bin/env bash
set -euo pipefail
cd /home/chx/projects/tokenflow-repro-paper-candidate
mode=${1:?baseline or tokenflow}
export PATH="$PWD/.venv/bin:/usr/local/cuda-12.1/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_VISIBLE_DEVICES=1
args=(--model-path "${MODEL_PATH:?fixed model path required}" --host 127.0.0.1 --port 30000 --tp-size 1 --dtype bfloat16 --mem-fraction-static "${MEM_FRACTION:-0.80}" --max-running-requests "${MAX_RUNNING:-128}" --attention-backend triton --sampling-backend pytorch --random-seed 42 --enable-cache-report --enable-deterministic-inference --disable-piecewise-cuda-graph)
if [[ "$mode" == tokenflow ]]; then
 export SGLANG_ENABLE_SYNC_CACHE=1 TOKENFLOW_REPRO_FENCE_WRITES=1
 export SGLANG_KV_OFFLOAD_POLICY_DEBUG_LOG=${POLICY_DEBUG:-0}
 args+=(--enable-hierarchical-cache --hicache-size 8 --kv-offload-policy paper_v1 --kv-offload-default-output-speed 20)
elif [[ "$mode" == baseline ]]; then
 export SGLANG_ENABLE_SYNC_CACHE=0
else exit 2
fi
if [[ "${NO_OVERLAP:-0}" == 1 ]]; then args+=(--disable-overlap-schedule); fi
printf '%q ' sglang serve "${args[@]}" > "repro/stage3-diagnostic-v1/env/${RUN_TAG:-$mode}-command.txt"
printf '\n' >> "repro/stage3-diagnostic-v1/env/${RUN_TAG:-$mode}-command.txt"
exec sglang serve "${args[@]}"
