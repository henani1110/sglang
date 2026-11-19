#!/usr/bin/env bash
set -euo pipefail

LABEL=${1:?unique label required}
PROBE=${2:?probe flag required}
MAX_SECONDS=${3:-220}
EPISODES_PER_ARM=${4:-8}
[[ "$PROBE" == 0 || "$PROBE" == 1 ]]

P=/home/chx/project/vl-stage-split-20261106-191642
S=/media/chx/Win10/vl-stage-split-20261106-191642
OLD=/home/chx/project/vl-encode-pilot-20261106-085428
SSD_OLD=/media/chx/Win10/vl-encode-pilot-20261106-085428
PY="$OLD/env/serve311/bin/python"
GPU_UUID=GPU-7d51d535-c0d2-577a-2a08-d6fbdd9b93a8
CONT=/home/chx/project/vl-stage-split-20261106-191642/continuation

mkdir -p "$S/results" "$S/logs"
test -x "$PY"
test -d "$S/datasets"
test -d "$SSD_OLD/models/Qwen3-VL-2B-Instruct"
if ss -ltnp | grep -q ':8271 '; then
  echo "port 8271 already occupied" >&2
  exit 10
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU_UUID" \
    | tr -d ' ' | grep -v -x '2966' | grep -q .; then
  echo "GPU0 has a non-baseline compute process" >&2
  exit 11
fi

LEDGER="$CONT/logs/resource-ledger.csv"
if [[ ! -f "$LEDGER" ]]; then
  printf 'label,probe,start_utc,stop_utc,service_seconds,ready,client_exit\n' > "$LEDGER"
fi
# MAX_SECONDS is a per-batch watchdog only. If it interrupts a long batch,
# preserve the checkpoint and resume in another uniquely labeled batch.

START_EPOCH=$(date +%s)
START_ISO=$(date -Is)
printf '%s,default,%s,start\n' "$START_ISO" "$LABEL" >> "$P/logs/service-ledger.csv"
nvidia-smi -q > "$S/logs/${LABEL}-gpu-pcie-start.txt"

if [[ "$PROBE" == 1 ]]; then
  nohup setsid env VL_PILOT_ASYNC_LOG=1 VL_PILOT_ASYNC_QUEUE=65536 \
    "$P/scripts/run_server.sh" default "$PROBE" "$LABEL" \
    > "$P/logs/${LABEL}-server.log" 2>&1 < /dev/null &
else
  nohup setsid "$P/scripts/run_server.sh" default "$PROBE" "$LABEL" \
    > "$P/logs/${LABEL}-server.log" 2>&1 < /dev/null &
fi
SPID=$!
PGID=$(ps -o pgid= -p "$SPID" | tr -d ' ')
printf '%s\n' "$SPID" > "$P/logs/${LABEL}-server.pid"
printf '%s\n' "$PGID" > "$P/logs/${LABEL}-server.pgid"

nohup setsid "$P/scripts/monitor_resources.sh" "$SPID" "$MAX_SECONDS" \
  "$S/logs/${LABEL}-resources.csv" \
  > "$P/logs/${LABEL}-monitor.log" 2>&1 < /dev/null &
MPID=$!
printf '%s\n' "$MPID" > "$P/logs/${LABEL}-monitor.pid"

READY=0
for _ in $(seq 1 150); do
  if curl -fsS http://127.0.0.1:8271/health >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$SPID" 2>/dev/null; then
    break
  fi
  sleep 2
done

CLIENT_RC=99
if [[ "$READY" == 1 ]]; then
  nvidia-smi -q > "$S/logs/${LABEL}-gpu-pcie-ready.txt"
  set +e
  "$PY" "$P/scripts/campaign_a.py" \
    --base-url http://127.0.0.1:8271 \
    --model-path "$SSD_OLD/models/Qwen3-VL-2B-Instruct" \
    --image-dir "$S/datasets" \
    --control-file "$P/pilot_control.json" \
    --client-events "$S/logs/${LABEL}-client-events.jsonl" \
    --diagnostics-file "$S/logs/${LABEL}-diagnostics.jsonl" \
    --equivalence-file "$S/logs/${LABEL}-equivalence.jsonl" \
    --results-file "$S/results/${LABEL}.json" \
    --cell "A-${LABEL}" \
    --probe "$PROBE" \
    --episodes-per-arm "$EPISODES_PER_ARM" \
    > "$P/logs/${LABEL}-client.stdout.log" 2>&1
  CLIENT_RC=$?
  set -e
else
  echo "server_failed_to_become_ready_within_300s" >> "$P/logs/${LABEL}-client.stdout.log"
fi

SIGNAL_ISO=$(date -Is)
printf '%s,%s,graceful_signal\n' "$SIGNAL_ISO" "$SPID" >> "$P/logs/service-ledger.csv"
kill -TERM "$SPID" 2>/dev/null || true
for _ in $(seq 1 180); do
  if ! kill -0 "$SPID" 2>/dev/null; then break; fi
  sleep 1
done
if kill -0 "$SPID" 2>/dev/null; then
  kill -TERM -- "-$PGID" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! kill -0 "$SPID" 2>/dev/null; then break; fi
    sleep 1
  done
fi
if kill -0 "$SPID" 2>/dev/null; then
  kill -KILL -- "-$PGID" 2>/dev/null || true
fi

for _ in $(seq 1 10); do
  if ! kill -0 "$MPID" 2>/dev/null; then break; fi
  sleep 1
done
kill -TERM "$MPID" 2>/dev/null || true

LEFTOVER=$(ps -eo pid=,pgid=,stat=,cmd= | awk -v g="$PGID" '$2==g && $3 !~ /^Z/ {print}')
if [[ -n "$LEFTOVER" ]]; then
  echo "service process group remains: $LEFTOVER" >&2
  exit 12
fi
for _ in $(seq 1 10); do
  GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_UUID" | head -1 | tr -d ' ')
  if [[ "$GPU_MEM" -le 1000 ]] && ! ss -ltnp | grep -q ':8271 '; then break; fi
  sleep 2
done
if [[ "$GPU_MEM" -gt 1000 ]] || ss -ltnp | grep -q ':8271 '; then
  echo "post-stop resource verification failed: gpu0=${GPU_MEM}MiB" >&2
  exit 13
fi

STOP_ISO=$(date -Is)
STOP_EPOCH=$(date +%s)
DURATION=$((STOP_EPOCH - START_EPOCH))
printf '%s,%s,stop\n' "$STOP_ISO" "$SPID" >> "$P/logs/service-ledger.csv"
printf '%s,%s,%s,%s,%s,%s,%s\n' \
  "$LABEL" "$PROBE" "$(date -d "$START_ISO" -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$(date -d "$STOP_ISO" -u +%Y-%m-%dT%H:%M:%SZ)" "$DURATION" "$READY" "$CLIENT_RC" >> "$LEDGER"
printf 'label=%s\nprobe=%s\nclient_exit=%s\nstart=%s\nsignal=%s\nstop=%s\nservice_seconds=%s\ngpu0_memory_mib=%s\nport8271=free\n' \
  "$LABEL" "$PROBE" "$CLIENT_RC" "$START_ISO" "$SIGNAL_ISO" "$STOP_ISO" "$DURATION" "$GPU_MEM" \
  > "$P/logs/${LABEL}-runner-status.txt"
[[ "$READY" == 1 && "$CLIENT_RC" == 0 ]]
