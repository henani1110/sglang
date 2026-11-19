#!/usr/bin/env bash
set -euo pipefail

STAGE=${1:?stage: calibration or formal}
POLICY=${2:?policy: N, F, S, A, or U}
GRID=${3:?image grid: 512 or 768}
LANES=${4:?active text streams: 2 or 6}
ARRIVAL=${5:?arrival: uniform or burst}
BLOCK=${6:?block: 0, 1, or 2}
ATTEMPT=${7:-a0}
SEED_OFFSETS=("${@:8}")
if [[ ${#SEED_OFFSETS[@]} -eq 0 ]]; then
  SEED_OFFSETS=(0 1 2 3)
fi
for offset in "${SEED_OFFSETS[@]}"; do
  [[ "$offset" =~ ^[0-3]$ ]]
done

RUN_ID=vl-stage-split-20261106-191642
P=/home/chx/project/$RUN_ID
S=/media/chx/Win10/$RUN_ID
D_P=$P/continuation/D
D_S=$S/continuation/D
OLD=/home/chx/project/vl-encode-pilot-20261106-085428
SSD_OLD=/media/chx/Win10/vl-encode-pilot-20261106-085428
PYTHON=$OLD/env/serve311/bin/python
MODEL=$SSD_OLD/models/Qwen3-VL-2B-Instruct
PORT=8271
GPU0_UUID=GPU-7d51d535-c0d2-577a-2a08-d6fbdd9b93a8

[[ "$STAGE" == calibration || "$STAGE" == formal ]]
[[ "$POLICY" =~ ^(N|F|S|A|U)$ ]]
[[ "$GRID" == 512 || "$GRID" == 768 ]]
[[ "$LANES" == 2 || "$LANES" == 6 ]]
[[ "$ARRIVAL" == uniform || "$ARRIVAL" == burst ]]
[[ "$BLOCK" =~ ^[0-2]$ ]]
if [[ "$STAGE" == calibration && "$POLICY" == U ]]; then
  echo "U is not part of calibration" >&2
  exit 2
fi
if [[ "$POLICY" == U && "$ARRIVAL" != burst ]]; then
  echo "U is only part of the paired-burst matrix" >&2
  exit 2
fi

mkdir -p "$D_S/logs/$STAGE" "$D_S/results/$STAGE" "$D_S/manifests"
SESSION="D${STAGE}-${POLICY}-G${GRID}-L${LANES}-${ARRIVAL}-B$((BLOCK + 1))-${ATTEMPT}"
SERVER_LOG="$D_S/logs/$STAGE/${SESSION}-server-stdout.log"
SERVER_EVENTS="$D_S/logs/$STAGE/${SESSION}-server-events.jsonl"
SCHEDULER_EVENTS="$D_S/logs/$STAGE/${SESSION}-scheduler-events.jsonl"
RESOURCE_LOG="$D_S/logs/resource-ledger.log"
SERVER_PID=""
MONITOR_PID=""

append_resource() {
  {
    printf '===== %s stage=%s session=%s event=%s =====\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$STAGE" "$SESSION" "$1"
    nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv,noheader,nounits
    nvidia-smi -q -i "$GPU0_UUID" | grep -A14 'GPU Link Info' | head -15 || true
    df -h /media/chx/Win10
  } >> "$RESOURCE_LOG" 2>&1
}

cleanup() {
  if [[ -n "$MONITOR_PID" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    append_resource "server_stop_requested"
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  append_resource "session_released"
}
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)$PORT$"; then
  append_resource "preflight_refused_port_conflict"
  echo "port $PORT is already listening; refuse to attach to an unknown server" >&2
  exit 3
fi
[[ -x "$PYTHON" ]] || { echo "vLLM environment Python is missing: $PYTHON" >&2; exit 3; }
[[ -d "$MODEL" ]] || { echo "fixed model path is missing: $MODEL" >&2; exit 3; }
if [[ "$STAGE" == calibration ]]; then
  for index in 00 01 02 03; do
    [[ -s "$D_S/datasets/geometry-$GRID/geometry-$index.png" ]] || {
      echo "calibration geometry control is missing: $D_S/datasets/geometry-$GRID/geometry-$index.png" >&2
      exit 3
    }
  done
else
  [[ -s "$D_S/datasets/holdout-final/holdout-selected-manifest.json" ]] || {
    echo "formal holdout manifest is missing from the SSD dataset directory" >&2
    exit 3
  }
fi
append_resource "session_start"
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

export VL_PILOT_EVENT_FILE="$SERVER_EVENTS"
export VL_PILOT_SCHED_FILE="$SCHEDULER_EVENTS"
bash "$D_P/scripts/run_server_d.sh" "$POLICY" "$SESSION" 1 "$PORT" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
READY_DEADLINE=$((SECONDS + 300))
while ! curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM process exited before readiness; see $SERVER_LOG" >&2
    exit 4
  fi
  if (( SECONDS >= READY_DEADLINE )); then
    echo "vLLM did not become ready within the per-attempt diagnostic window; preserve logs and inspect before retry" >&2
    exit 5
  fi
  sleep 5
done
append_resource "server_ready"

(
  while kill -0 "$SERVER_PID" 2>/dev/null; do
    append_resource "periodic_sample"
    sleep 30
  done
) &
MONITOR_PID=$!

for offset in "${SEED_OFFSETS[@]}"; do
  if [[ "$STAGE" == calibration ]]; then
    SEED=$((300 + offset))
    RUN="Dcal-${POLICY}-G${GRID}-L${LANES}-${ARRIVAL}-s${SEED}-${ATTEMPT}"
    IMAGE_DIR="$D_S/datasets/geometry-$GRID"
    RESULT="$D_S/results/calibration/$RUN.json"
    CLIENT="$D_S/logs/calibration/$RUN-client-events.jsonl"
  else
    SEED=$((1000 + BLOCK * 4 + offset))
    RUN="D-${POLICY}-G${GRID}-L${LANES}-${ARRIVAL}-s${SEED}-${ATTEMPT}"
    IMAGE_DIR=""
    RESULT="$D_S/results/formal/$RUN.json"
    CLIENT="$D_S/logs/formal/$RUN-client-events.jsonl"
  fi
  mkdir -p "$(dirname "$RESULT")" "$(dirname "$CLIENT")"
  COMMON=(--base-url "http://127.0.0.1:$PORT" --model-name qwen3vlpilot
    --model-path "$MODEL" --image-size "$GRID" --text-streams "$LANES"
    --arrival-mode "$ARRIVAL" --policy "$POLICY" --seed "$SEED" --block "$((BLOCK + 1))"
    --run-id "$RUN" --control-file "$D_P/pilot_control.json" --client-events "$CLIENT"
    --results-file "$RESULT")
  if [[ "$STAGE" == calibration ]]; then
    append_resource "run_start:$RUN"
    set +e
    "$PYTHON" "$D_P/scripts/campaign_d.py" "${COMMON[@]}" \
      --control-image-dir "$IMAGE_DIR" --warmup-seconds 10 --measure-seconds 30 \
      --calibration-diagnostic | tee "$D_S/logs/calibration/$RUN-console.jsonl"
    CAMPAIGN_RC=${PIPESTATUS[0]}
    set -e
    /usr/bin/python3 - "$D_S/manifests/calibration-run-index.json" "$RUN" "$POLICY" "$GRID" "$LANES" "$ARRIVAL" "$SEED" "$RESULT" "$CLIENT" "$SCHEDULER_EVENTS" "$SERVER_EVENTS" "$CAMPAIGN_RC" <<'PY'
import json, os, pathlib, sys
path=pathlib.Path(sys.argv[1]); path.parent.mkdir(parents=True,exist_ok=True)
payload=json.loads(path.read_text()) if path.exists() else {"schema_version":1,"runs":[]}
result_path=pathlib.Path(sys.argv[8])
if result_path.exists(): result=json.loads(result_path.read_text())
else:
    result={"status":"failed","run_id":sys.argv[2],"error":"campaign process exited before writing result JSON"}
    result_path.parent.mkdir(parents=True,exist_ok=True); result_path.write_text(json.dumps(result,indent=2)+"\n")
entry={"run_id":sys.argv[2],"policy":sys.argv[3],"image_size":int(sys.argv[4]),"text_streams":int(sys.argv[5]),"arrival_mode":sys.argv[6],"seed":int(sys.argv[7]),"status":result.get("status"),"attempt":sys.argv[2].rsplit("-",1)[-1],"result_path":sys.argv[8],"client_events_path":sys.argv[9],"scheduler_events_path":sys.argv[10],"server_events_path":sys.argv[11],"process_exit":int(sys.argv[12])}
if not any(row.get("run_id")==entry["run_id"] for row in payload["runs"]): payload["runs"].append(entry)
tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n"); os.replace(tmp,path)
PY
    append_resource "run_end:$RUN:exit=$CAMPAIGN_RC"
    if [[ "$CAMPAIGN_RC" != 0 ]]; then
      exit 10
    fi
  else
    append_resource "run_start:$RUN"
    set +e
    "$PYTHON" "$D_P/scripts/campaign_d.py" "${COMMON[@]}" \
      --dataset-manifest "$D_S/datasets/holdout-final/holdout-selected-manifest.json" \
      --warmup-seconds 60 --measure-seconds 180 | tee "$D_S/logs/formal/$RUN-console.jsonl"
    CAMPAIGN_RC=${PIPESTATUS[0]}
    set -e
    /usr/bin/python3 - "$D_S/manifests/d-run-manifest.json" "$RUN" "$POLICY" "$GRID" "$LANES" "$ARRIVAL" "$SEED" "$((BLOCK + 1))" "$RESULT" "$CLIENT" "$SCHEDULER_EVENTS" "$SERVER_EVENTS" "$CAMPAIGN_RC" "$ATTEMPT" <<'PY'
import json, os, pathlib, sys
path=pathlib.Path(sys.argv[1]); payload=json.loads(path.read_text())
result_path=pathlib.Path(sys.argv[9])
if result_path.exists(): result=json.loads(result_path.read_text())
else:
    result={"status":"failed","run_id":sys.argv[2],"error":"campaign process exited before writing result JSON"}
    result_path.parent.mkdir(parents=True,exist_ok=True); result_path.write_text(json.dumps(result,indent=2)+"\n")
entry={"run_id":sys.argv[2],"policy":sys.argv[3],"image_size":int(sys.argv[4]),"text_streams":int(sys.argv[5]),"arrival_mode":sys.argv[6],"seed":int(sys.argv[7]),"block":int(sys.argv[8]),"status":result.get("status"),"attempt":sys.argv[14],"result_path":sys.argv[9],"client_events_path":sys.argv[10],"scheduler_events_path":sys.argv[11],"server_events_path":sys.argv[12],"process_exit":int(sys.argv[13]),"error":result.get("error")}
payload.setdefault("matrix",{}).setdefault("seed_records",[]).append(entry)
successful_keys={(row["image_size"],row["text_streams"],row["arrival_mode"],row["policy"],row["seed"]) for row in payload["matrix"]["seed_records"] if row.get("status")=="completed"}
payload["matrix"]["completed"]=len(successful_keys)
payload["matrix"]["invalid_attempts"]=sum(1 for row in payload["matrix"]["seed_records"] if row.get("status")!="completed" or int(row.get("process_exit",0))!=0)
tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n"); os.replace(tmp,path)
PY
    append_resource "run_end:$RUN:exit=$CAMPAIGN_RC"
    if [[ "$CAMPAIGN_RC" != 0 ]]; then
      exit 10
    fi
  fi
done

append_resource "session_runs_complete"
