#!/usr/bin/env bash
set -euo pipefail

CLIENT_PID="$1"
OUTPUT="$2"
mkdir -p "$(dirname "$OUTPUT")"
if [[ ! -s "$OUTPUT" ]]; then
  printf '%s\n' 'timestamp_epoch,gpu_index,uuid,utilization_gpu_percent,memory_used_mib,sm_clock_mhz,memory_clock_mhz,temperature_c,power_w,pstate,pci_link_speed,pci_link_width' > "$OUTPUT"
fi

while kill -0 "$CLIENT_PID" 2>/dev/null; do
  TIMESTAMP=$(date +%s.%N)
  while IFS= read -r ROW; do
    GPU_INDEX=${ROW%%,*}
    GPU_INDEX=$(printf '%s' "$GPU_INDEX" | xargs)
    case "$GPU_INDEX" in
      0) DEVICE=0000:01:00.0 ;;
      1) DEVICE=0000:05:00.0 ;;
      *) DEVICE=unknown ;;
    esac
    if [[ "$DEVICE" == unknown ]]; then
      SPEED=unknown
      WIDTH=unknown
    else
      SPEED=$(cat "/sys/bus/pci/devices/$DEVICE/current_link_speed" 2>/dev/null || printf unknown)
      WIDTH=$(cat "/sys/bus/pci/devices/$DEVICE/current_link_width" 2>/dev/null || printf unknown)
    fi
    printf '%s,%s,%s,%s\n' "$TIMESTAMP" "$ROW" "$SPEED" "$WIDTH" >> "$OUTPUT"
  done < <(nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw,pstate --format=csv,noheader,nounits)
  sleep 5
done
