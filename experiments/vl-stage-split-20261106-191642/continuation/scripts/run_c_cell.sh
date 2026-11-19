#!/usr/bin/env bash
set -euo pipefail

LABEL=${1:?unique run label required}
STRATEGY=${2:?strategy required: N, F, S, or U}
shift 2
PROBE=1
MAX_SECONDS=1800
EPISODES=1
TEXT_STREAMS=2
IMAGE_SIZE=768
PHASE=measurement
BLOCK=c-block1
TRIAL_START=0
TRIAL_IDS=
ATTEMPT_ID=0
PREFILL_INTERVAL=1
LM_BUDGET=2048
ENCODER_BUDGET=0
TEXT_TOKENS=192
ARRIVAL_OFFSET_MS=5
CLIENT_SCRIPT=campaign_c.py
while (($#)); do
  case "$1" in
    --probe) PROBE=$2; shift 2 ;;
    --max-seconds) MAX_SECONDS=$2; shift 2 ;;
    --episodes) EPISODES=$2; shift 2 ;;
    --text-streams) TEXT_STREAMS=$2; shift 2 ;;
    --image-size) IMAGE_SIZE=$2; shift 2 ;;
    --phase) PHASE=$2; shift 2 ;;
    --block) BLOCK=$2; shift 2 ;;
    --trial-start) TRIAL_START=$2; shift 2 ;;
    --trial-ids) TRIAL_IDS=$2; shift 2 ;;
    --attempt-id) ATTEMPT_ID=$2; shift 2 ;;
    --prefill-interval) PREFILL_INTERVAL=$2; shift 2 ;;
    --lm-budget) LM_BUDGET=$2; shift 2 ;;
    --encoder-budget) ENCODER_BUDGET=$2; shift 2 ;;
    --text-tokens) TEXT_TOKENS=$2; shift 2 ;;
    --arrival-offset-ms) ARRIVAL_OFFSET_MS=$2; shift 2 ;;
    --client-script) CLIENT_SCRIPT=$2; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
if [[ -n "$TRIAL_IDS" ]]; then
  IFS=',' read -r -a TRIAL_ARRAY <<< "$TRIAL_IDS"
  EPISODES=${#TRIAL_ARRAY[@]}
  TRIAL_START=${TRIAL_ARRAY[0]}
else
  TRIAL_IDS=$(seq -s, "$TRIAL_START" "$((TRIAL_START + EPISODES - 1))")
fi

[[ "$STRATEGY" == N || "$STRATEGY" == F || "$STRATEGY" == S || "$STRATEGY" == U ]]
[[ "$PROBE" == 0 || "$PROBE" == 1 ]]
[[ "$TEXT_STREAMS" == 2 || "$TEXT_STREAMS" == 6 ]]
[[ "$IMAGE_SIZE" == 512 || "$IMAGE_SIZE" == 768 ]]
[[ "$EPISODES" =~ ^[0-9]+$ && "$EPISODES" -gt 0 ]]

P=/home/chx/project/vl-stage-split-20261106-191642
S=/media/chx/Win10/vl-stage-split-20261106-191642
OLD=/home/chx/project/vl-encode-pilot-20261106-085428
SSD_OLD=/media/chx/Win10/vl-encode-pilot-20261106-085428
PY="$OLD/env/serve311/bin/python"
GPU_UUID=GPU-7d51d535-c0d2-577a-2a08-d6fbdd9b93a8
CONT="$P/continuation"
IMAGE_DIR="$S/datasets-$IMAGE_SIZE"
mkdir -p "$S/results" "$S/logs" "$CONT/logs"
test -x "$PY"
test -d "$IMAGE_DIR"
test -d "$SSD_OLD/models/Qwen3-VL-2B-Instruct"
for idx in 00 01 02 03; do test -s "$IMAGE_DIR/geometry-$idx.png"; done
if ss -ltnp | grep -q ':8271 '; then
  echo "port 8271 already occupied" >&2
  exit 10
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU_UUID" \
    | tr -d ' ' | grep -v -x '2966' | grep -q .; then
  echo "GPU0 has a non-baseline compute process" >&2
  exit 11
fi

LEDGER="$CONT/logs/C-run-ledger.csv"
if [[ ! -f "$LEDGER" ]]; then
  printf 'label,strategy,phase,block,trial_start,trial_ids,episodes,image_size,text_streams,prefill_interval,lm_budget,encoder_budget,start_utc,stop_utc,service_seconds,ready,client_exit\n' > "$LEDGER"
fi
START_EPOCH=$(date +%s)
START_ISO=$(date -Is)
printf '%s,default,%s,start\n' "$START_ISO" "$LABEL" >> "$P/logs/service-ledger.csv"
nvidia-smi -q > "$S/logs/${LABEL}-gpu-pcie-start.txt"

nohup setsid env VL_PILOT_ASYNC_LOG=1 VL_PILOT_ASYNC_QUEUE=65536 \
  "$CONT/scripts/run_server_c.sh" "$STRATEGY" "$LABEL" "$PROBE" \
  "$PREFILL_INTERVAL" "$LM_BUDGET" "$ENCODER_BUDGET" \
  > "$P/logs/${LABEL}-server.log" 2>&1 < /dev/null &
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
  if curl -fsS http://127.0.0.1:8271/health >/dev/null 2>&1; then READY=1; break; fi
  if ! kill -0 "$SPID" 2>/dev/null; then break; fi
  sleep 2
done
CLIENT_RC=99
if [[ "$READY" == 1 ]]; then
  nvidia-smi -q > "$S/logs/${LABEL}-gpu-pcie-ready.txt"
  set +e
  "$PY" "$CONT/scripts/$CLIENT_SCRIPT" \
    --base-url http://127.0.0.1:8271 \
    --model-path "$SSD_OLD/models/Qwen3-VL-2B-Instruct" \
    --image-dir "$IMAGE_DIR" \
    --control-file "$P/pilot_control.json" \
    --client-events "$S/logs/${LABEL}-client-events.jsonl" \
    --diagnostics-file "$S/logs/${LABEL}-diagnostics.jsonl" \
    --equivalence-file "$S/logs/${LABEL}-equivalence.jsonl" \
    --results-file "$S/results/${LABEL}.json" \
    --cell "$LABEL" --strategy "$STRATEGY" --phase "$PHASE" --block "$BLOCK" \
    --trial-start "$TRIAL_START" --trial-ids "$TRIAL_IDS" --attempt-id "$ATTEMPT_ID" --episodes "$EPISODES" \
    --text-streams "$TEXT_STREAMS" --text-tokens "$TEXT_TOKENS" \
    --image-size "$IMAGE_SIZE" --arrival-offset-ms "$ARRIVAL_OFFSET_MS" \
    --prefill-interval "$PREFILL_INTERVAL" --lm-budget "$LM_BUDGET" \
    --encoder-budget "$ENCODER_BUDGET" \
    > "$P/logs/${LABEL}-client.stdout.log" 2>&1
  CLIENT_RC=$?
  set -e
else
  echo "server_failed_to_become_ready_within_300s" >> "$P/logs/${LABEL}-client.stdout.log"
fi

SIGNAL_ISO=$(date -Is)
printf '%s,%s,graceful_signal\n' "$SIGNAL_ISO" "$SPID" >> "$P/logs/service-ledger.csv"
kill -TERM "$SPID" 2>/dev/null || true
for _ in $(seq 1 180); do if ! kill -0 "$SPID" 2>/dev/null; then break; fi; sleep 1; done
if kill -0 "$SPID" 2>/dev/null; then
  kill -TERM -- "-$PGID" 2>/dev/null || true
  for _ in $(seq 1 30); do if ! kill -0 "$SPID" 2>/dev/null; then break; fi; sleep 1; done
fi
if kill -0 "$SPID" 2>/dev/null; then kill -KILL -- "-$PGID" 2>/dev/null || true; fi
for _ in $(seq 1 10); do if ! kill -0 "$MPID" 2>/dev/null; then break; fi; sleep 1; done
kill -TERM "$MPID" 2>/dev/null || true

LEFTOVER=$(ps -eo pid=,pgid=,stat=,cmd= | awk -v g="$PGID" '$2==g && $3 !~ /^Z/ {print}')
if [[ -n "$LEFTOVER" ]]; then echo "service process group remains: $LEFTOVER" >&2; exit 12; fi
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
printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
  "$LABEL" "$STRATEGY" "$PHASE" "$BLOCK" "$TRIAL_START" "${TRIAL_IDS//,/;}" "$EPISODES" \
  "$IMAGE_SIZE" "$TEXT_STREAMS" "$PREFILL_INTERVAL" "$LM_BUDGET" "$ENCODER_BUDGET" \
  "$(date -d "$START_ISO" -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$(date -d "$STOP_ISO" -u +%Y-%m-%dT%H:%M:%SZ)" "$DURATION" "$READY" "$CLIENT_RC" >> "$LEDGER"
printf 'label=%s\nstrategy=%s\nphase=%s\nclient_exit=%s\nstart=%s\nsignal=%s\nstop=%s\nservice_seconds=%s\ngpu0_memory_mib=%s\nport8271=free\n' \
  "$LABEL" "$STRATEGY" "$PHASE" "$CLIENT_RC" "$START_ISO" "$SIGNAL_ISO" "$STOP_ISO" "$DURATION" "$GPU_MEM" \
  > "$P/logs/${LABEL}-runner-status.txt"
[[ "$READY" == 1 && "$CLIENT_RC" == 0 ]]
