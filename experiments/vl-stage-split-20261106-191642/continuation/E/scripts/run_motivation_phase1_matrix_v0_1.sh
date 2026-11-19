#!/usr/bin/env bash
set -u

RUN_ID="vl-stage-split-20261106-191642"
PROJECT_DIR="/home/chx/project/${RUN_ID}/continuation/E"
SSD_DIR="/media/chx/Win10/${RUN_ID}/continuation/E"
PYTHON_BIN="/home/chx/project/vl-encode-pilot-20261106-085428/env/serve311/bin/python"
MANAGER_SCRIPT="${PROJECT_DIR}/scripts/run_motivation_phase1_v0_1.py"
BLOCK_ID="motivation-phase1-v0.1"

batch_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
log_root="${SSD_DIR}/logs/formal-online-v0.1/${BLOCK_ID}"
run_root="${SSD_DIR}/runs/formal-online-v0.1/${BLOCK_ID}"
mkdir -p "${log_root}"
matrix_log="${log_root}/matrix-${batch_id}.log"
attempts_tsv="${log_root}/matrix-${batch_id}-attempts.tsv"
printf 'seed\ttext_lanes\tarrival_mode\tattempt\texit_code\trun_log\n' > "${attempts_tsv}"

attempt_count=0
nonzero_count=0

log_line() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${matrix_log}"
}

run_one() {
  local seed="$1"
  local lanes="$2"
  local arrival_mode="$3"
  local attempt=0
  local run_name="E-N-G768-L${lanes}-${arrival_mode}-s${seed}-a${attempt}"
  local output_dir="${run_root}/${run_name}"
  local run_log="${log_root}/${run_name}.log"
  local exit_code

  attempt_count=$((attempt_count + 1))
  log_line "START seed=${seed} lanes=${lanes} mode=${arrival_mode} attempt=${attempt}"
  if [[ -e "${output_dir}" ]]; then
    exit_code=125
    log_line "PRESERVED_EXISTING_OUTPUT seed=${seed} lanes=${lanes} mode=${arrival_mode} path=${output_dir}"
  else
    "${PYTHON_BIN}" "${MANAGER_SCRIPT}" \
      --project "${PROJECT_DIR}" \
      --ssd "${SSD_DIR}" \
      --python "${PYTHON_BIN}" \
      --arm N \
      --seed "${seed}" \
      --attempt "${attempt}" \
      --edge 768 \
      --text-lanes "${lanes}" \
      --arrival-mode "${arrival_mode}" \
      --warmup-seconds 60 \
      --measure-seconds 180 \
      --port 8285 \
      --block-id "${BLOCK_ID}" > "${run_log}" 2>&1
    exit_code=$?
    log_line "FINISH seed=${seed} lanes=${lanes} mode=${arrival_mode} attempt=${attempt} exit_code=${exit_code} run_log=${run_log}"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${seed}" "${lanes}" "${arrival_mode}" "${attempt}" "${exit_code}" "${run_log}" \
    >> "${attempts_tsv}"
  if [[ "${exit_code}" -ne 0 ]]; then
    nonzero_count=$((nonzero_count + 1))
  fi
}

run_one 41001 2 none
run_one 41001 2 burst
run_one 41001 6 burst
run_one 41001 6 none

run_one 41002 2 burst
run_one 41002 2 none
run_one 41002 6 none
run_one 41002 6 burst

run_one 41003 2 none
run_one 41003 2 burst
run_one 41003 6 burst
run_one 41003 6 none

run_one 41004 6 none
run_one 41004 6 burst
run_one 41004 2 burst
run_one 41004 2 none

run_one 41005 6 burst
run_one 41005 6 none
run_one 41005 2 none
run_one 41005 2 burst

run_one 41006 6 none
run_one 41006 6 burst
run_one 41006 2 burst
run_one 41006 2 none

log_line "MATRIX_FINISHED attempts=${attempt_count} nonzero=${nonzero_count} attempts_tsv=${attempts_tsv}"
[[ "${nonzero_count}" -eq 0 ]]
