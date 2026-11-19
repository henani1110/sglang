#!/usr/bin/env bash
set -Eeuo pipefail

WORKTREE=/home/chx/projects/sglang-h3-exact-lora-20261029
SCRIPT_DIR="$WORKTREE/experiments/minimax-h3-sm89-kernel-20261030/scripts"
PYTHON=/home/chx/projects/sglang/.venv/bin/python
RESULT_ROOT="$WORKTREE/experiments/minimax-h3-sm89-kernel-20261030/results/x16-fc2-exact-gate-fusion-20261031"
MICROBENCH="$RESULT_ROOT/real-activation-microbench.json"
SERVER_PID=

if [[ -e "$RESULT_ROOT" ]]; then
  echo "refusing to overwrite existing result directory: $RESULT_ROOT" >&2
  exit 2
fi
mkdir -p "$RESULT_ROOT/logs"

GPU_WIDTH=$(cat /sys/bus/pci/devices/0000:01:00.0/current_link_width)
GPU_MAX_WIDTH=$(cat /sys/bus/pci/devices/0000:01:00.0/max_link_width)
GPU_SPEED=$(cat /sys/bus/pci/devices/0000:01:00.0/current_link_speed)
AVAILABLE_GB=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
if [[ "$GPU_WIDTH" != 16 || "$GPU_MAX_WIDTH" != 16 ]]; then
  echo "GPU0 is not PCIe x16: current=$GPU_WIDTH max=$GPU_MAX_WIDTH" >&2
  exit 3
fi
if (( AVAILABLE_GB < 100 )); then
  echo "insufficient /home free space: $AVAILABLE_GB G" >&2
  exit 4
fi
if curl -fsS --max-time 2 http://127.0.0.1:30014/health >/dev/null 2>&1; then
  echo "port 30014 is already occupied; refusing to interfere" >&2
  exit 5
fi

cat > "$RESULT_ROOT/preflight.json" <<EOF
{
  "hostname": "$(hostname)",
  "started_at": "$(date -Is)",
  "gpu0_pci_address": "0000:01:00.0",
  "gpu0_current_link_width": "$GPU_WIDTH",
  "gpu0_max_link_width": "$GPU_MAX_WIDTH",
  "gpu0_current_idle_link_speed": "$GPU_SPEED",
  "home_available_gb": $AVAILABLE_GB,
  "worktree_head": "$(git -C "$WORKTREE" rev-parse HEAD)",
  "worktree_branch": "$(git -C "$WORKTREE" branch --show-current)"
}
EOF
git -C "$WORKTREE" status --short > "$RESULT_ROOT/worktree-status-before.txt"

stop_server() {
  [[ -n "$SERVER_PID" ]] || return 0
  if [[ ! -d "/proc/$SERVER_PID" ]]; then
    SERVER_PID=
    return 0
  fi
  local cwd cmd
  cwd=$(readlink -f "/proc/$SERVER_PID/cwd" 2>/dev/null || true)
  cmd=$(tr '\0' ' ' < "/proc/$SERVER_PID/cmdline" 2>/dev/null || true)
  if [[ "$cwd" != "$WORKTREE" || "$cmd" != *"sglang"* || "$cmd" != *"--port 30014"* ]]; then
    echo "refusing to stop PID $SERVER_PID because its identity changed: cwd=$cwd cmd=$cmd" >&2
    return 1
  fi
  kill -TERM "$SERVER_PID"
  for _ in $(seq 1 120); do
    if [[ ! -d "/proc/$SERVER_PID" ]]; then
      SERVER_PID=
      return 0
    fi
    sleep 5
  done
  echo "server PID $SERVER_PID did not exit after SIGTERM" >&2
  return 1
}
trap stop_server EXIT INT TERM

echo "starting Sage2 service on GPU0 x16" | tee -a "$RESULT_ROOT/logs/orchestrator.log"
env \
  SGLANG_EXACT_LORA_GATE_MICROBENCH=1 \
  SGLANG_EXACT_LORA_GATE_MICROBENCH_PATH="$MICROBENCH" \
  nohup bash "$SCRIPT_DIR/start-x16-sage2.sh" \
  > "$RESULT_ROOT/logs/sage2-service.log" 2>&1 < /dev/null &
SERVER_PID=$!

deadline=$((SECONDS + 1800))
while (( SECONDS < deadline )); do
  if curl -fsS --max-time 5 http://127.0.0.1:30014/health >/dev/null 2>&1; then
    echo "Sage2 service ready pid=$SERVER_PID" | tee -a "$RESULT_ROOT/logs/orchestrator.log"
    break
  fi
  if [[ ! -d "/proc/$SERVER_PID" ]]; then
    tail -100 "$RESULT_ROOT/logs/sage2-service.log" >&2
    echo "Sage2 service exited before becoming healthy" >&2
    exit 6
  fi
  sleep 15
done
if ! curl -fsS --max-time 5 http://127.0.0.1:30014/health >/dev/null 2>&1; then
  tail -100 "$RESULT_ROOT/logs/sage2-service.log" >&2
  echo "Sage2 service did not become healthy within 30 minutes" >&2
  exit 7
fi

"$PYTHON" "$SCRIPT_DIR/run_x16_exact_lora_gate_fusion.py" warmups \
  2>&1 | tee "$RESULT_ROOT/logs/warmups.log"

if [[ ! -s "$MICROBENCH" ]]; then
  echo "exact FC2 gate fusion did not emit its real-activation microbenchmark" >&2
  exit 8
fi
"$PYTHON" - "$MICROBENCH" <<'PY'
import json
import statistics
import sys

data = json.load(open(sys.argv[1]))
reference = data["per_round_ms_per_call"]["reference_two_kernels_ms"]
candidate = data["per_round_ms_per_call"]["fused_one_kernel_ms"]
reductions = [100.0 * (1.0 - new / old) for old, new in zip(reference, candidate)]
median = statistics.median(reductions)
print(json.dumps({"round_reductions_pct": reductions, "median_reduction_pct": median}, indent=2))
if not all(value > 0 for value in reductions) or median < 2.0:
    raise SystemExit("no stable >=2% microbenchmark gain; stopping before E2E")
PY

GPU_WIDTH=$(cat /sys/bus/pci/devices/0000:01:00.0/current_link_width)
if [[ "$GPU_WIDTH" != 16 ]]; then
  echo "GPU0 lost PCIe x16 link before formal samples: current=$GPU_WIDTH" >&2
  exit 9
fi

"$PYTHON" "$SCRIPT_DIR/run_x16_exact_lora_gate_fusion.py" formal \
  2>&1 | tee "$RESULT_ROOT/logs/formal-abba.log"
"$PYTHON" "$SCRIPT_DIR/run_x16_exact_lora_gate_fusion.py" summary \
  2>&1 | tee "$RESULT_ROOT/logs/summary.log"
stop_server

if curl -fsS --max-time 2 http://127.0.0.1:30014/health >/dev/null 2>&1; then
  echo "port 30014 is still serving after cleanup" >&2
  exit 10
fi
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used,utilization.gpu --format=csv \
  > "$RESULT_ROOT/nvidia-smi-after.csv"
echo "completed_at=$(date -Is)" | tee -a "$RESULT_ROOT/logs/orchestrator.log"
