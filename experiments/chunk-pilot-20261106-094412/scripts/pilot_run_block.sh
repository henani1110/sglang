#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
    echo "usage: $0 BLOCK GPU_LABEL GPU_UUID CHUNK FIRST_TRACE SECOND_TRACE [SMOKE]" >&2
    exit 2
fi

block=$1
gpu_label=$2
gpu_uuid=$3
chunk=$4
first_trace=$5
second_trace=$6
smoke=${7:-0}
root=/home/chx/project/chunk-pilot-20261106-094412
python=/home/chx/projects/sglang/.venv/bin/python
started=$(date +%s)
budget_started=$(cat "$root/logs/budget.started_epoch")
if (( started - budget_started >= 7200 )); then
    echo "120 GPU-minute wall budget reached before $block" >&2
    exit 124
fi

server_log="$root/logs/$block-server.log"
server_info="$root/logs/$block-server-info.json"
setsid "$root/scripts/pilot_launch_server.sh" "$gpu_uuid" "$chunk" 30000 "$server_log" >/dev/null 2>&1 &
server_pid=$!
echo "$server_pid" > "$root/logs/$block-server.pid"
date -Is > "$root/logs/$block-started_at"

stop_server() {
    if kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM -- "-$server_pid" 2>/dev/null || true
        for _ in {1..20}; do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 1
        done
    fi
    wait "$server_pid" 2>/dev/null || true
    date -Is > "$root/logs/$block-stopped_at"
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

deadline=$((started + 180))
ready=0
while (( $(date +%s) < deadline )); do
    if curl --max-time 2 -fsS http://127.0.0.1:30000/health > "$root/logs/$block-health.txt" 2>&1; then
        ready=1
        break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "$block server exited before readiness" >&2
        tail -100 "$server_log" >&2
        exit 1
    fi
    sleep 2
done
if (( ! ready )); then
    echo "$block server failed the 180-second readiness gate" >&2
    tail -100 "$server_log" >&2
    exit 2
fi

date -Is > "$root/logs/$block-ready_at"
curl --max-time 10 -fsS http://127.0.0.1:30000/server_info > "$server_info"
"$python" - "$server_info" "$chunk" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    actual = json.load(stream)
expected = {
    "chunked_prefill_size": int(sys.argv[2]),
    "dtype": "bfloat16",
    "kv_cache_dtype": "bfloat16",
    "context_length": 8192,
    "max_running_requests": 8,
    "max_total_tokens": 32768,
    "max_prefill_tokens": 8192,
    "prefill_max_requests": 8,
    "mem_fraction_static": 0.75,
    "schedule_policy": "fcfs",
    "attention_backend": "triton",
    "decode_attention_backend": "triton",
    "disable_radix_cache": True,
    "tp_size": 1,
}
for key, value in expected.items():
    if actual.get(key) != value:
        raise SystemExit(f"actual {key}={actual.get(key)!r}, expected {value!r}")
for phase in ("prefill", "decode"):
    backend = actual["cuda_graph_config"][phase]["backend"]
    if backend != "disabled":
        raise SystemExit(f"actual CUDA graph {phase}={backend!r}")
print(json.dumps({key: actual.get(key) for key in expected}, sort_keys=True))
PY

echo "$block ready on $gpu_label chunk=$chunk at $(cat "$root/logs/$block-ready_at")"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader

if [[ "$smoke" == "1" ]]; then
    smoke_dir="$root/runs/$block-smoke"
    timeout 180s env CUDA_VISIBLE_DEVICES= "$python" "$root/scripts/pilot_replay.py" run --smoke --trace "$root/traces/frozen/uniform.json" --output "$smoke_dir" --timeout 90 --no-progress-timeout 30 > "$smoke_dir.runner.log" 2>&1
    cat "$smoke_dir.runner.log"
    "$python" - "$smoke_dir/run.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    run = json.load(stream)
if run["requests"] != 4 or run["passed"] != 4:
    raise SystemExit(f"smoke did not pass all requests: {run}")
PY
fi

for trace in "$first_trace" "$second_trace"; do
    now=$(date +%s)
    if (( now - budget_started >= 7200 )); then
        echo "120 GPU-minute wall budget reached during $block" >&2
        exit 124
    fi
    trace_file="$root/traces/frozen/$trace.json"
    run_dir="$root/runs/$block-$trace"
    timeout 180s env CUDA_VISIBLE_DEVICES= "$python" "$root/scripts/pilot_replay.py" run --trace "$trace_file" --output "$run_dir" --timeout 90 --no-progress-timeout 30 > "$run_dir.runner.log" 2>&1
    cat "$run_dir.runner.log"
    "$python" - "$run_dir/run.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    run = json.load(stream)
if run["requests"] != 40 or run["passed"] != 40:
    raise SystemExit(f"run did not pass all requests: {run}")
PY
done

now=$(date +%s)
if (( now - budget_started >= 7200 )); then
    echo "120 GPU-minute wall budget reached during $block" >&2
    exit 124
fi
