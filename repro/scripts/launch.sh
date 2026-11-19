#!/usr/bin/env bash
set -euo pipefail
cd /home/chx/projects/tokenflow-repro
mode=${1:?baseline or tokenflow}
export PATH="$PWD/.venv/bin:/usr/local/cuda-12.1/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_VISIBLE_DEVICES=1
args=(--model-path /home/chx/models/Qwen3-0.6B --host 127.0.0.1 --port 30000 --tp-size 1 --dtype bfloat16 --mem-fraction-static 0.80 --max-running-requests "${MAX_RUNNING:-8}" --attention-backend triton --sampling-backend pytorch --random-seed 42)
if [[ "$mode" == tokenflow ]]; then
  export SGLANG_KV_OFFLOAD_POLICY_DEBUG_LOG=${POLICY_DEBUG:-0}
  args+=(--enable-hierarchical-cache --hicache-size 8 --kv-offload-policy paper_v1 --kv-offload-default-output-speed 20)
elif [[ "$mode" != baseline ]]; then exit 2
fi
if [[ -n "${OFFLOAD_INTERVAL:-}" ]]; then args+=(--kv-offload-reschedule-interval "$OFFLOAD_INTERVAL"); fi
if [[ -n "${MAX_TOTAL_TOKENS:-}" ]]; then args+=(--max-total-tokens "$MAX_TOTAL_TOKENS" --context-length 4096); fi
printf '%q ' sglang serve "${args[@]}" > "repro/env/$mode-command.txt"
printf '\n' >> "repro/env/$mode-command.txt"
exec sglang serve "${args[@]}"
