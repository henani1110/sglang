#!/usr/bin/env bash
set -u

RUN_ID=vl-stage-split-20261106-191642
E_P=/home/chx/project/$RUN_ID/continuation/E
E_S=/media/chx/Win10/$RUN_ID/continuation/E
OLD=/home/chx/project/vl-encode-pilot-20261106-085428
PY=$OLD/env/serve311/bin/python
IMAGE=$E_S/runs/processor-smoke/e-trace-v2-11-queue-assets/square-512.png
ACTIVE_SERVER_PID=
ACTIVE_LAUNCHER_PID=

cleanup_active() {
  if [[ -n "$ACTIVE_SERVER_PID" ]] && kill -0 "$ACTIVE_SERVER_PID" 2>/dev/null; then
    kill -INT "$ACTIVE_SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      if ! ss -ltn "sport = :8285" | grep -q LISTEN; then break; fi
      sleep 1
    done
  fi
  if [[ -n "$ACTIVE_LAUNCHER_PID" ]] && kill -0 "$ACTIVE_LAUNCHER_PID" 2>/dev/null; then
    kill -TERM "$ACTIVE_LAUNCHER_PID" 2>/dev/null || true
  fi
}
trap cleanup_active EXIT INT TERM

run_one() {
  block_id=$1
  profile=$2
  probe=$3
  queue_probe=$4
  label=e-overhead-${block_id}-${profile}
  launcher_log=$E_S/logs/$label-launcher.log
  client_log=$E_S/logs/$label-client.log
  output_jsonl=$E_S/runs/processor-smoke/$label-client.jsonl
  mkdir -p "$E_S/logs" "$E_S/runs/processor-smoke" "$E_S/configs"
  if ss -ltn "sport = :8285" | grep -q LISTEN; then
    echo "SKIP $label: port 8285 already in use" | tee -a "$E_S/logs/probe-overhead-runner.log"
    return 0
  fi

  cat > "$E_S/configs/$label.json" <<EOF
{"schema_version":1,"profile":"$profile","block_id":"$block_id","VL_PILOT_PROBE":"$probe","VL_PILOT_QUEUE_PROBE":"$queue_probe","warmups":2,"measurements":6,"seed":17701,"max_tokens":16,"image_sha256":"$(sha256sum "$IMAGE" | awk '{print $1}')"}
EOF
  cd "$E_P"
  env VL_PILOT_LABEL="$label" VL_PILOT_PROBE="$probe" VL_PILOT_QUEUE_PROBE="$queue_probe" \
    ./scripts/run_server_e.sh > "$launcher_log" 2>&1 < /dev/null &
  ACTIVE_LAUNCHER_PID=$!
  echo "$ACTIVE_LAUNCHER_PID" > "$E_S/logs/$label-launcher.pid"

  ready=0
  for _ in $(seq 1 120); do
    if curl -fsS http://127.0.0.1:8285/health >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$ACTIVE_LAUNCHER_PID" 2>/dev/null; then break; fi
    sleep 1
  done
  if [[ "$ready" != 1 ]]; then
    echo "START_FAILED $label" | tee -a "$E_S/logs/probe-overhead-runner.log"
    tail -50 "$launcher_log" >> "$E_S/logs/probe-overhead-runner.log" 2>/dev/null || true
    ACTIVE_SERVER_PID=
    if [[ -n "$ACTIVE_LAUNCHER_PID" ]] && kill -0 "$ACTIVE_LAUNCHER_PID" 2>/dev/null; then
      kill -TERM "$ACTIVE_LAUNCHER_PID" 2>/dev/null || true
    fi
    ACTIVE_LAUNCHER_PID=
    return 0
  fi
  ACTIVE_SERVER_PID=$(ps -eo pid,ppid,args | awk -v parent="$ACTIVE_LAUNCHER_PID" '$2 == parent && /vllm serve/ {print $1; exit}')
  if [[ -z "$ACTIVE_SERVER_PID" ]]; then
    echo "SERVER_PID_NOT_FOUND $label" | tee -a "$E_S/logs/probe-overhead-runner.log"
    if [[ -n "$ACTIVE_LAUNCHER_PID" ]] && kill -0 "$ACTIVE_LAUNCHER_PID" 2>/dev/null; then
      kill -INT "$ACTIVE_LAUNCHER_PID" 2>/dev/null || true
    fi
    return 0
  fi
  echo "$ACTIVE_SERVER_PID" > "$E_S/logs/$label-server.pid"
  echo "STARTED $label launcher=$ACTIVE_LAUNCHER_PID server=$ACTIVE_SERVER_PID" | tee -a "$E_S/logs/probe-overhead-runner.log"

  if ! "$PY" "$E_P/scripts/calibrate_queue_probe_overhead_e.py" \
      --profile "$profile" --block-id "$block_id" --image "$IMAGE" \
      --output-jsonl "$output_jsonl" --warmup-count 2 --sample-count 6 \
      --seed 17701 --max-tokens 16 > "$client_log" 2>&1; then
    echo "CLIENT_FAILED $label" | tee -a "$E_S/logs/probe-overhead-runner.log"
    tail -50 "$client_log" >> "$E_S/logs/probe-overhead-runner.log" 2>/dev/null || true
  else
    echo "CLIENT_DONE $label" | tee -a "$E_S/logs/probe-overhead-runner.log"
  fi

  if [[ -n "$ACTIVE_SERVER_PID" ]] && kill -0 "$ACTIVE_SERVER_PID" 2>/dev/null; then
    kill -INT "$ACTIVE_SERVER_PID" 2>/dev/null || true
  fi
  for _ in $(seq 1 45); do
    if ! ss -ltn "sport = :8285" | grep -q LISTEN; then break; fi
    sleep 1
  done
  if ss -ltn "sport = :8285" | grep -q LISTEN; then
    echo "PORT_STILL_OPEN $label" | tee -a "$E_S/logs/probe-overhead-runner.log"
    return 0
  fi
  ACTIVE_SERVER_PID=
  if [[ -n "$ACTIVE_LAUNCHER_PID" ]] && kill -0 "$ACTIVE_LAUNCHER_PID" 2>/dev/null; then
    wait "$ACTIVE_LAUNCHER_PID" 2>/dev/null || true
  fi
  ACTIVE_LAUNCHER_PID=
  nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader > "$E_S/logs/$label-gpu-after.csv"
  echo "STOPPED $label" | tee -a "$E_S/logs/probe-overhead-runner.log"
}

run_one b1 probe-off 0 0
run_one b1 queue-only 0 1
run_one b1 full-probe 1 1
run_one b2 full-probe 1 1
run_one b2 queue-only 0 1
run_one b2 probe-off 0 0

echo "CALIBRATION_SEQUENCE_DONE"
