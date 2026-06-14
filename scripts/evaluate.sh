#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PREPROCESS_IMAGE="${CRASHTWIN_PREPROCESS_IMAGE:-nuochen1203/crashtwin-preprocess:draft-20260614-runtime}"
RECONSTRUCT_IMAGE="${CRASHTWIN_RECONSTRUCT_IMAGE:-nuochen1203/crashtwin-reconstruct:draft-20260614-runtime}"
CACHE_DIR="${CRASHTWIN_CACHE_DIR:-${REPO_ROOT}/.cache}"

METHOD_NAME=""
PREDICTIONS=""
OUTPUT=""
BENCHMARK="${REPO_ROOT}/benchmark/crashtwin_eval.csv"
BENCHMARK_ROOT="${REPO_ROOT}"
CONFIG="${REPO_ROOT}/configs/default.yaml"
GPUS="0"
SKIP_PREPROCESS=0
SKIP_RECONSTRUCTION=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/evaluate.sh \
    --method-name <model_name> \
    --predictions predictions/<model_name> \
    --output outputs/<model_name> \
    --gpus 0,1,2,3

Required:
  --method-name       Name written to logs and output folders.
  --predictions       Folder containing one <video_id>.mp4 per benchmark row.
  --output            Output folder.

Optional:
  --benchmark         Benchmark CSV. Default: benchmark/crashtwin_eval.csv
  --benchmark-root    Folder containing benchmark/ and checkpoints/. Default: repo root
  --config            Evaluation config. Default: configs/default.yaml
  --gpus              Docker GPU device list, e.g. 0 or 0,1,2,3. Default: 0
  --skip-preprocess   Reuse existing preprocessed outputs.
  --skip-reconstruction
                      Reuse existing reconstruction outputs.
  --dry-run           Print docker commands without running them.
EOF
}

die() {
  echo "Error: $*" >&2
  exit 2
}

abspath() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    realpath -m "${path}"
  else
    realpath -m "${REPO_ROOT}/${path}"
  fi
}

container_path() {
  local host_path
  host_path="$(abspath "$1")"
  case "${host_path}" in
    "${REPO_ROOT}"/*) printf '/crashtwin/%s\n' "$(realpath --relative-to="${REPO_ROOT}" "${host_path}")" ;;
    "${REPO_ROOT}") printf '/crashtwin\n' ;;
    *) die "Path must be inside the CrashTwin repository because the repo is mounted at /crashtwin: ${host_path}" ;;
  esac
}

run_docker() {
  local host_gpus="$1"
  local image="$2"
  shift 2

  local gpu_arg
  if [[ "${host_gpus}" == "all" ]]; then
    gpu_arg="all"
  else
    gpu_arg="\"device=${host_gpus}\""
  fi

  local cmd=(
    docker run --rm
    --gpus "${gpu_arg}"
    --ipc host
    --shm-size 32g
    -e "NVIDIA_VISIBLE_DEVICES=${host_gpus}"
    -e CRASHTWIN_CACHE=/cache
    -v "${REPO_ROOT}:/crashtwin"
    -v "${CACHE_DIR}:/cache"
    -w /crashtwin
    "${image}"
    "$@"
  )

  printf '[crashtwin] '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  if [[ "${DRY_RUN}" == "0" ]]; then
    "${cmd[@]}"
  fi
}

launch_docker() {
  local log_file="$1"
  local host_gpus="$2"
  local image="$3"
  shift 3

  local gpu_arg
  if [[ "${host_gpus}" == "all" ]]; then
    gpu_arg="all"
  else
    gpu_arg="\"device=${host_gpus}\""
  fi

  local cmd=(
    docker run --rm
    --gpus "${gpu_arg}"
    --ipc host
    --shm-size 32g
    -e "NVIDIA_VISIBLE_DEVICES=${host_gpus}"
    -e CRASHTWIN_CACHE=/cache
    -v "${REPO_ROOT}:/crashtwin"
    -v "${CACHE_DIR}:/cache"
    -w /crashtwin
    "${image}"
    "$@"
  )

  printf '[crashtwin] log=%s ' "${log_file}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  if [[ "${DRY_RUN}" == "0" ]]; then
    "${cmd[@]}" >"${log_file}" 2>&1 &
  fi
}

make_shards() {
  local benchmark="$1"
  local shard_dir="$2"
  shift 2
  local gpus=("$@")
  local num_shards="${#gpus[@]}"

  mkdir -p "${shard_dir}"
  local header
  header="$(head -n 1 "${benchmark}")"
  for ((i = 0; i < num_shards; i++)); do
    printf '%s\n' "${header}" >"${shard_dir}/benchmark_shard_${i}.csv"
  done

  awk -v n="${num_shards}" -v dir="${shard_dir}" '
    NR == 1 { next }
    {
      shard = (NR - 2) % n
      print $0 >> dir "/benchmark_shard_" shard ".csv"
    }
  ' "${benchmark}"
}

wait_for_jobs() {
  local phase="$1"
  shift
  local logs=("$@")
  local failed=0

  for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  PIDS=()

  if [[ "${failed}" != "0" ]]; then
    echo "Error: ${phase} failed. Log files:" >&2
    printf '  %s\n' "${logs[@]}" >&2
    for log in "${logs[@]}"; do
      if [[ -f "${log}" ]]; then
        echo "---- tail ${log} ----" >&2
        tail -n 40 "${log}" >&2
      fi
    done
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method-name) METHOD_NAME="${2:-}"; shift 2 ;;
    --predictions) PREDICTIONS="${2:-}"; shift 2 ;;
    --output) OUTPUT="${2:-}"; shift 2 ;;
    --benchmark) BENCHMARK="$(abspath "${2:-}")"; shift 2 ;;
    --benchmark-root) BENCHMARK_ROOT="$(abspath "${2:-}")"; shift 2 ;;
    --config) CONFIG="$(abspath "${2:-}")"; shift 2 ;;
    --gpus) GPUS="${2:-}"; shift 2 ;;
    --skip-preprocess) SKIP_PREPROCESS=1; shift ;;
    --skip-reconstruction) SKIP_RECONSTRUCTION=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ -n "${METHOD_NAME}" ]] || die "--method-name is required"
[[ -n "${PREDICTIONS}" ]] || die "--predictions is required"
[[ -n "${OUTPUT}" ]] || die "--output is required"
[[ -n "${GPUS}" ]] || die "--gpus cannot be empty"

PREDICTIONS="$(abspath "${PREDICTIONS}")"
OUTPUT="$(abspath "${OUTPUT}")"
BENCHMARK="$(abspath "${BENCHMARK}")"
BENCHMARK_ROOT="$(abspath "${BENCHMARK_ROOT}")"
CONFIG="$(abspath "${CONFIG}")"

[[ -d "${PREDICTIONS}" ]] || die "Prediction folder not found: ${PREDICTIONS}"
[[ -f "${BENCHMARK}" ]] || die "Benchmark CSV not found: ${BENCHMARK}"
[[ -f "${CONFIG}" ]] || die "Config not found: ${CONFIG}"
[[ -d "${BENCHMARK_ROOT}/benchmark/auto_json" ]] || die "Missing benchmark/auto_json under ${BENCHMARK_ROOT}"
[[ -d "${BENCHMARK_ROOT}/benchmark/vehicle_specs" ]] || die "Missing benchmark/vehicle_specs under ${BENCHMARK_ROOT}"
[[ -d "${BENCHMARK_ROOT}/checkpoints" ]] || die "Missing checkpoints under ${BENCHMARK_ROOT}"

mkdir -p "${OUTPUT}/per_video" "${CACHE_DIR}"

IFS=',' read -r -a GPU_LIST <<<"${GPUS}"
if [[ "${GPUS}" == "all" ]]; then
  GPU_LIST=("all")
fi

SHARD_DIR="${OUTPUT}/_shards"
LOG_DIR="${OUTPUT}/logs"
mkdir -p "${SHARD_DIR}" "${LOG_DIR}"
make_shards "${BENCHMARK}" "${SHARD_DIR}" "${GPU_LIST[@]}"

PIDS=()

if [[ "${SKIP_PREPROCESS}" == "0" ]]; then
  PREPROCESS_LOGS=()
  for index in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[$index]}"
    log_file="${LOG_DIR}/preprocess_gpu_${gpu}.log"
    PREPROCESS_LOGS+=("${log_file}")
    launch_docker "${log_file}" "${gpu}" "${PREPROCESS_IMAGE}" \
      python3 /crashtwin/scripts/container_preprocess.py \
      --inputs "$(container_path "${PREDICTIONS}")" \
      --benchmark "$(container_path "${SHARD_DIR}/benchmark_shard_${index}.csv")" \
      --benchmark-root "$(container_path "${BENCHMARK_ROOT}")" \
      --output "$(container_path "${OUTPUT}/per_video")" \
      --config "$(container_path "${CONFIG}")" \
      --gpus "0"
    if [[ "${DRY_RUN}" == "0" ]]; then
      PIDS+=("$!")
    fi
  done
  wait_for_jobs "preprocess" "${PREPROCESS_LOGS[@]}"
fi

if [[ "${SKIP_RECONSTRUCTION}" == "0" ]]; then
  RECONSTRUCT_LOGS=()
  for index in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[$index]}"
    log_file="${LOG_DIR}/reconstruct_gpu_${gpu}.log"
    RECONSTRUCT_LOGS+=("${log_file}")
    launch_docker "${log_file}" "${gpu}" "${RECONSTRUCT_IMAGE}" \
      python3 /crashtwin/scripts/container_reconstruct.py \
      --benchmark "$(container_path "${SHARD_DIR}/benchmark_shard_${index}.csv")" \
      --benchmark-root "$(container_path "${BENCHMARK_ROOT}")" \
      --per-video-dir "$(container_path "${OUTPUT}/per_video")" \
      --config "$(container_path "${CONFIG}")" \
      --gpus "0"
    if [[ "${DRY_RUN}" == "0" ]]; then
      PIDS+=("$!")
    fi
  done
  wait_for_jobs "reconstruction" "${RECONSTRUCT_LOGS[@]}"
fi

run_docker "${GPU_LIST[0]}" "${RECONSTRUCT_IMAGE}" \
  python3 /crashtwin/scripts/container_score.py \
  --benchmark "$(container_path "${BENCHMARK}")" \
  --per-video-dir "$(container_path "${OUTPUT}/per_video")" \
  --output "$(container_path "${OUTPUT}")"

echo "Summary: ${OUTPUT}/summary_metrics.csv"
