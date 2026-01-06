#!/usr/bin/env bash
set -euo pipefail

if [[ "${VERBOSE:-0}" == "1" ]]; then
  set -x
fi

usage() {
  cat <<'EOF'
Usage:
  bash experiments/what-if/collect_dataset_costmatrix.sh <run_dir|run_name> <num_instances> [num_shards]

Collect a what-if node-removal dataset for cost-matrix TSP instances (MatNet/ATSP),
then merge + summarize.

Environment variables (optional):
  SEED                  Default: 0
  TAG                   Default: costmat_v1
  CONCORDE_TIMEOUT_SEC  Default: 60
  OVERWRITE             Default: 0
  MAX_REMOVED_NODES     Default: "" (unset means all nodes)
  LOG_EVERY_SEC         Default: 15
  PYTHON                Default: python
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_ARG="$1"
NUM_INSTANCES="$2"
NUM_SHARDS="${3:-1}"

PYTHON="${PYTHON:-python}"
SEED="${SEED:-0}"
TAG="${TAG:-costmat_v1}"
CONCORDE_TIMEOUT_SEC="${CONCORDE_TIMEOUT_SEC:-60}"
OVERWRITE="${OVERWRITE:-0}"
MAX_REMOVED_NODES="${MAX_REMOVED_NODES:-}"
LOG_EVERY_SEC="${LOG_EVERY_SEC:-15}"

if [[ -d "${RUN_ARG}" ]]; then
  RUN_DIR="${RUN_ARG}"
elif [[ -d "${REPO_ROOT}/runs/${RUN_ARG}" ]]; then
  RUN_DIR="${REPO_ROOT}/runs/${RUN_ARG}"
else
  echo "ERROR: Could not resolve run dir from '${RUN_ARG}'"
  exit 1
fi

RUN_NAME="$(basename "${RUN_DIR}")"
DATASET_ID="${RUN_NAME}__n${NUM_INSTANCES}__seed${SEED}"
if [[ -n "${TAG}" ]]; then
  DATASET_ID="${DATASET_ID}__${TAG}"
fi

PROCESSED_ROOT="${PROCESSED_ROOT:-${REPO_ROOT}/experiments/what-if/data/processed}"
DATA_DIR="${PROCESSED_ROOT}/${DATASET_ID}"
RAW_DIR="${DATA_DIR}/raw"

mkdir -p "${RAW_DIR}"

echo "[costmatrix] Run dir:    ${RUN_DIR}"
echo "[costmatrix] Instances:  ${NUM_INSTANCES}"
echo "[costmatrix] Shards:     ${NUM_SHARDS}"
echo "[costmatrix] Dataset id: ${DATASET_ID}"
echo "[costmatrix] Data dir:   ${DATA_DIR}"

COLLECT_PY="${REPO_ROOT}/experiments/what-if/collect/collect_dataset_costmatrix.py"
if [[ ! -f "${COLLECT_PY}" ]]; then
  echo "ERROR: Missing collector: ${COLLECT_PY}"
  exit 1
fi

for shard_idx in $(seq 0 $((NUM_SHARDS-1))); do
  shard_path="${RAW_DIR}/shard_$(printf '%04d' "${shard_idx}").pt"
  if [[ -f "${shard_path}" && "${OVERWRITE}" != "1" ]]; then
    echo "[costmatrix] Shard exists, skipping: ${shard_path}"
    continue
  fi
  echo "[costmatrix] Collect shard ${shard_idx}/${NUM_SHARDS} -> ${shard_path}"
  cmd=(
    "${PYTHON}" "${COLLECT_PY}"
    --run_dir "${RUN_DIR}"
    --out_path "${shard_path}"
    --num_instances "${NUM_INSTANCES}"
    --num_shards "${NUM_SHARDS}"
    --shard_idx "${shard_idx}"
    --seed "${SEED}"
    --concorde_timeout_sec "${CONCORDE_TIMEOUT_SEC}"
    --log_every_sec "${LOG_EVERY_SEC}"
  )
  if [[ "${OVERWRITE}" == "1" ]]; then
    cmd+=(--overwrite)
  fi
  if [[ -n "${MAX_REMOVED_NODES}" ]]; then
    cmd+=(--max_removed_nodes "${MAX_REMOVED_NODES}")
  fi
  "${cmd[@]}"
done

echo "[costmatrix] Merge shards -> dataset.pt"
"${PYTHON}" "${REPO_ROOT}/experiments/what-if/collect/merge_shards.py" --raw_dir "${RAW_DIR}" --out_dir "${DATA_DIR}"
echo "[costmatrix] Summarize"
"${PYTHON}" "${REPO_ROOT}/experiments/what-if/collect/summarize_dataset.py" --data_dir "${DATA_DIR}"
echo "[costmatrix] Done: ${DATA_DIR}"

