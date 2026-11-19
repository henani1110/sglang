#!/usr/bin/env bash
set -euo pipefail
PID=${1:?server pid}
MAX_SECONDS=${2:?remaining stage seconds}
OUT=${3:?csv path}
GPU_UUID=GPU-7d51d535-c0d2-577a-2a08-d6fbdd9b93a8
BASELINE_PID=2966
PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
START=$(date +%s)
printf 'time_iso,elapsed_s,gpu_memory_mib,gpu_util_percent,gpu_power_w,process_group_rss_kib,host_mem_available_b\n' > "$OUT"
while kill -0 "$PID" 2>/dev/null; do
  NOW=$(date +%s)
  ELAPSED=$((NOW-START))
  GPU=$(nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits -i "$GPU_UUID" | head -1 | tr -d ' ')
  MEM=$(printf '%s' "$GPU" | cut -d, -f1)
  UTIL=$(printf '%s' "$GPU" | cut -d, -f2)
  POWER=$(printf '%s' "$GPU" | cut -d, -f3)
  RSS=$(ps -eo pgid=,rss= | awk -v p="$PGID" '$1==p {s+=$2} END {print s+0}')
  AVAILABLE=$(awk '/MemAvailable:/ {print $2*1024}' /proc/meminfo)
  printf '%s,%s,%s,%s,%s,%s,%s\n' "$(date -Is)" "$ELAPSED" "$MEM" "$UTIL" "$POWER" "$RSS" "$AVAILABLE" >> "$OUT"
  REASON=
  if [ "$MEM" -ge 20480 ]; then REASON=gpu_memory_cap; fi
  if [ "$RSS" -ge 50331648 ]; then REASON=process_group_rss_cap; fi
  if [ "$ELAPSED" -ge "$MAX_SECONDS" ]; then REASON=stage_gpu_time_budget; fi
  while read -r OTHER_PID; do
    [ -z "$OTHER_PID" ] && continue
    [ "$OTHER_PID" = "$BASELINE_PID" ] && continue
    OTHER_PGID=$(ps -o pgid= -p "$OTHER_PID" 2>/dev/null | tr -d ' ')
    if [ -n "$OTHER_PGID" ] && [ "$OTHER_PGID" != "$PGID" ]; then REASON=external_gpu_process; break; fi
  done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU_UUID" 2>/dev/null | tr -d ' ')
  if [ -n "$REASON" ]; then
    printf '%s,STOP,%s\n' "$(date -Is)" "$REASON" >> "$OUT"
    kill -TERM -- "-$PGID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true
    exit 0
  fi
  sleep 2
done
printf '%s,EXIT,server_process_ended\n' "$(date -Is)" >> "$OUT"
