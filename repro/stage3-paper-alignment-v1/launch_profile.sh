#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/chx/projects/tokenflow-repro-paper-candidate
cd "$ROOT"
export PATH="$ROOT/.venv/bin:/usr/local/cuda-12.1/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_VISIBLE_DEVICES=1
export SGLANG_KV_OFFLOAD_POLICY_DEBUG_LOG=0
unset MAX_TOTAL_TOKENS OFFLOAD_INTERVAL TOKENFLOW_REPRO_TRACE TOKENFLOW_REPRO_VERIFY_KV TOKENFLOW_REPRO_SNAPSHOT_WRITES

profile=${1:?usage: launch_profile.sh sglang-fork-plain|sglang-fork-chunked-2048|tokenflow-paper-v1-chunked-2048}
args=(
  --model-path "$ROOT/repro/models/Qwen2-7B-Instruct"
  --host 127.0.0.1
  --port 30000
  --tp-size 1
  --dtype bfloat16
  --mem-fraction-static 0.8
  --max-running-requests 128
  --max-total-tokens 77001
  --attention-backend triton
  --sampling-backend pytorch
  --random-seed 42
  --enable-cache-report
  --enable-deterministic-inference
  --disable-piecewise-cuda-graph
)

case "$profile" in
  sglang-fork-plain)
    export SGLANG_ENABLE_SYNC_CACHE=0
    unset TOKENFLOW_REPRO_FENCE_WRITES
    args+=(--chunked-prefill-size -1)
    ;;
  sglang-fork-chunked-2048)
    export SGLANG_ENABLE_SYNC_CACHE=0
    unset TOKENFLOW_REPRO_FENCE_WRITES
    args+=(--chunked-prefill-size 2048)
    ;;
  tokenflow-paper-v1-chunked-2048)
    export SGLANG_ENABLE_SYNC_CACHE=1
    export TOKENFLOW_REPRO_FENCE_WRITES=1
    args+=(
      --chunked-prefill-size 2048
      --enable-hierarchical-cache
      --hicache-size 8
      --kv-offload-policy paper_v1
      --kv-offload-default-output-speed 20
    )
    ;;
  *)
    echo "Unknown profile: $profile" >&2
    exit 2
    ;;
esac

mkdir -p repro/stage3-paper-alignment-v1/preflight
printf '%q ' sglang serve "${args[@]}" > "repro/stage3-paper-alignment-v1/preflight/${profile}-command.txt"
printf '\n' >> "repro/stage3-paper-alignment-v1/preflight/${profile}-command.txt"
exec sglang serve "${args[@]}"
