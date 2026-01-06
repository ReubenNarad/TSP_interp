#!/usr/bin/env bash
set -euo pipefail

# "Big" MatNet training run on an OSM road-network pool (precomputed KxK cost matrix).
#
# This uses the best config from quick sweeps on Seattle road-time pools:
# - embed_dim=256, layers=3, heads=8
# - batch_size=256 (stable and fast)
# - pool loaded into RAM (`--pool_in_memory`) for faster slicing
#
# Usage:
#   POOL_DIR=data/osm_pools/seattle_time_k10000_seed0 RUN_NAME=matnet_seattle_big bash scripts/train_matnet_pool_big.sh
#
# You can override any env var below.

RUN_NAME="${RUN_NAME:-matnet_seattle_pool_big_$(date +%Y%m%d_%H%M%S)}"
POOL_DIR="${POOL_DIR:-data/osm_pools/seattle_time_k10000_seed0}"

# Problem / data
NUM_LOC="${NUM_LOC:-100}"
SEED="${SEED:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-500}"
NUM_INSTANCES="${NUM_INSTANCES:-20000}" # instances per epoch
NUM_VAL="${NUM_VAL:-512}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"         # multi-workers were slower for pool slicing on this setup
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-200}"
# Scaling the cost matrix improves MatNet numerical stability on real road-time costs.
# This does not change the optimal tour (constant factor), and plotting scripts convert back to seconds.
COST_SCALE="${COST_SCALE:-1000.0}"

SAVE_RESULTS_EVERY="${SAVE_RESULTS_EVERY:-0}"
RESULTS_EVAL_BS="${RESULTS_EVAL_BS:-64}"

# Optim
LR="${LR:-1e-4}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-10}"
CLIP_VAL="${CLIP_VAL:-1.0}"
LR_DECAY="${LR_DECAY:-none}"            # try "cosine" for long runs
MIN_LR="${MIN_LR:-1e-6}"
EXP_GAMMA="${EXP_GAMMA:-}"              # optional (0<gamma<1) if LR_DECAY=exponential

# MatNet architecture
EMBED_DIM="${EMBED_DIM:-256}"
N_ENCODER_LAYERS="${N_ENCODER_LAYERS:-3}"
NUM_HEADS="${NUM_HEADS:-8}"
NORMALIZATION="${NORMALIZATION:-instance}"
TANH_CLIPPING="${TANH_CLIPPING:-10.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"

if [[ ! -d "${POOL_DIR}" ]]; then
  echo "ERROR: POOL_DIR does not exist: ${POOL_DIR}" >&2
  exit 1
fi

EXP_ARGS=()
if [[ -n "${EXP_GAMMA}" ]]; then
  EXP_ARGS+=(--exp_gamma "${EXP_GAMMA}")
fi

python -m policy.train_matnet \
  --run_name "${RUN_NAME}" \
  --pool_dir "${POOL_DIR}" \
  --pool_in_memory \
  --num_epochs "${NUM_EPOCHS}" \
  --num_instances "${NUM_INSTANCES}" \
  --num_val "${NUM_VAL}" \
  --num_loc "${NUM_LOC}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --log_every_n_steps "${LOG_EVERY_N_STEPS}" \
  --cost_scale "${COST_SCALE}" \
  --save_results_every "${SAVE_RESULTS_EVERY}" \
  --results_eval_batch_size "${RESULTS_EVAL_BS}" \
  --seed "${SEED}" \
  --lr "${LR}" \
  --checkpoint_freq "${CHECKPOINT_FREQ}" \
  --clip_val "${CLIP_VAL}" \
  --lr_decay "${LR_DECAY}" \
  --min_lr "${MIN_LR}" \
  "${EXP_ARGS[@]}" \
  --embed_dim "${EMBED_DIM}" \
  --n_encoder_layers "${N_ENCODER_LAYERS}" \
  --num_heads "${NUM_HEADS}" \
  --normalization "${NORMALIZATION}" \
  --tanh_clipping "${TANH_CLIPPING}" \
  --temperature "${TEMPERATURE}"

echo ""
echo "Run saved to: runs/${RUN_NAME}"
echo "Optional: solve Concorde baseline (val batch, symmetric only):"
echo "  python -m env.solve_costmatrix_with_concorde --run_name ${RUN_NAME} --max_instances 128 --output_name baseline_concorde_128.pkl --symmetrize none"
echo "Optional: generate train plot:"
echo "  python -m policy.plot_train_curve --run_name ${RUN_NAME} --baseline baseline_concorde_128.pkl --step 1"
echo "Optional: render sanity GIF:"
echo "  python -m policy.eval_matnet --run_name ${RUN_NAME} --pbf ../../GEPA_TSP/data/osm/Seattle.osm.pbf --num_epochs ${NUM_EPOCHS} --step 10"
