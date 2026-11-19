#!/usr/bin/env bash
set -Eeuo pipefail

WORKTREE=/home/chx/projects/sglang-h3-exact-lora-20261029
SCRIPT_DIR="$WORKTREE/experiments/minimax-h3-sm89-kernel-20261030/scripts"
PYTHON=/home/chx/projects/sglang/.venv/bin/python
RESULT_ROOT="$WORKTREE/experiments/minimax-h3-sm89-kernel-20261030/results/x16-five-arm-attribution-20261031"
TRACE_ROOT="$RESULT_ROOT/traces"
SERVER_PID=

mkdir -p "$RESULT_ROOT/logs" "$TRACE_ROOT"
if [[ -e "$RESULT_ROOT/summary.json" ]]; then
  echo "refusing to overwrite existing summary: $RESULT_ROOT/summary.json" >&2
  exit 2
fi

GPU_WIDTH=$(cat /sys/bus/pci/devices/0000:01:00.0/current_link_width)
GPU_MAX_WIDTH=$(cat /sys/bus/pci/devices/0000:01:00.0/max_link_width)
GPU_SPEED=$(cat /sys/bus/pci/devices/0000:01:00.0/current_link_speed)
GPU_MAX_SPEED=$(cat /sys/bus/pci/devices/0000:01:00.0/max_link_speed)
AVAILABLE_GB=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
if [[ "$GPU_WIDTH" != 16 || "$GPU_MAX_WIDTH" != 16 ]]; then
  echo "GPU0 is not PCIe x16: current=$GPU_WIDTH max=$GPU_MAX_WIDTH" >&2
  exit 3
fi
if (( AVAILABLE_GB < 100 )); then
  echo "insufficient /home free space for experiment: ${AVAILABLE_GB}G" >&2
  exit 4
fi
if curl -fsS --max-time 2 http://127.0.0.1:30014/health >/dev/null 2>&1; then
  echo "port 30014 already has a service; refusing to interfere" >&2
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
  "gpu0_max_link_speed": "$GPU_MAX_SPEED",
  "home_available_gb": $AVAILABLE_GB,
  "worktree_head": "$(git -C "$WORKTREE" rev-parse HEAD)",
  "worktree_branch": "$(git -C "$WORKTREE" branch --show-current)"
}
EOF

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
    echo "refusing to stop PID $SERVER_PID because process identity changed: cwd=$cwd cmd=$cmd" >&2
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
  echo "server PID $SERVER_PID did not exit after SIGTERM; leaving it running" >&2
  return 1
}
trap stop_server EXIT INT TERM

start_server() {
  local backend=$1 launcher=$2
  if curl -fsS --max-time 2 http://127.0.0.1:30014/health >/dev/null 2>&1; then
    echo "port 30014 became occupied before $backend startup" >&2
    return 1
  fi
  echo "starting $backend service on GPU0..." | tee -a "$RESULT_ROOT/logs/orchestrator.log"
  env SGLANG_TORCH_PROFILER_DIR="$TRACE_ROOT" \
    nohup bash "$SCRIPT_DIR/$launcher" \
    > "$RESULT_ROOT/logs/$backend-service.log" 2>&1 < /dev/null &
  SERVER_PID=$!
  local deadline=$((SECONDS + 1800))
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 5 http://127.0.0.1:30014/health > /dev/null 2>&1; then
      echo "$backend ready pid=$SERVER_PID" | tee -a "$RESULT_ROOT/logs/orchestrator.log"
      return 0
    fi
    if [[ ! -d "/proc/$SERVER_PID" ]]; then
      tail -100 "$RESULT_ROOT/logs/$backend-service.log" >&2
      echo "$backend service exited before becoming healthy" >&2
      return 1
    fi
    sleep 15
  done
  tail -100 "$RESULT_ROOT/logs/$backend-service.log" >&2
  echo "$backend service did not become healthy within 30 minutes" >&2
  return 1
}

stop_server
start_server fa start-x16-fa-baseline.sh
"$PYTHON" "$SCRIPT_DIR/run_x16_five_arm_attribution.py" fa \
  2>&1 | tee "$RESULT_ROOT/logs/fa-suite.log"
stop_server

start_server sage2 start-x16-sage2.sh
"$PYTHON" "$SCRIPT_DIR/run_x16_five_arm_attribution.py" sage2 \
  2>&1 | tee "$RESULT_ROOT/logs/sage2-suite-and-profiles.log"
stop_server

echo "completed_at=$(date -Is)" | tee -a "$RESULT_ROOT/logs/orchestrator.log"
