#!/usr/bin/env bash
set -euo pipefail

# Long MatNet training run on an OSM road-network pool (precomputed KxK cost matrix).
#
# Usage:
#   bash scripts/train_matnet_pool_long.sh
#
# Override any hyperparameter by setting env vars, e.g.:
#   RUN_NAME=matnet_seattle_long NUM_EPOCHS=200 BATCH_SIZE=512 bash scripts/train_matnet_pool_long.sh

RUN_NAME="${RUN_NAME:-matnet_seattle_pool_n100_long_$(date +%Y%m%d_%H%M%S)}"

# Pool must exist (build it with `python -m road_tsp.build_pool ...`).
POOL_DIR="${POOL_DIR:-data/osm_pools/seattle_time_k2000_seed0_demo}"

# Problem / data
NUM_LOC="${NUM_LOC:-100}"
SEED="${SEED:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-200}"
NUM_INSTANCES="${NUM_INSTANCES:-20000}" # instances per epoch (RL4CO: train_data_size)
NUM_VAL="${NUM_VAL:-256}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-50}"
COST_SCALE="${COST_SCALE:-1000.0}"

# Optim
LR="${LR:-1e-4}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-10}"
CLIP_VAL="${CLIP_VAL:-1.0}"
LR_DECAY="${LR_DECAY:-none}"      # none|cosine|linear|exponential
MIN_LR="${MIN_LR:-1e-6}"          # used for cosine/linear and for exponential when EXP_GAMMA unset
EXP_GAMMA="${EXP_GAMMA:-}"        # optional per-epoch exponential factor (0<gamma<1)

# MatNet architecture
EMBED_DIM="${EMBED_DIM:-128}"
N_ENCODER_LAYERS="${N_ENCODER_LAYERS:-5}"
NUM_HEADS="${NUM_HEADS:-8}"
NORMALIZATION="${NORMALIZATION:-instance}" # batch|instance|layer|none
TANH_CLIPPING="${TANH_CLIPPING:-10.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"

if [[ ! -d "${POOL_DIR}" ]]; then
  echo "ERROR: POOL_DIR does not exist: ${POOL_DIR}" >&2
  echo "Build a pool first, e.g.:" >&2
  echo "  python -m road_tsp.build_pool --pbf ../../GEPA_TSP/data/osm/Seattle.osm.pbf --bbox 47.58,47.64,-122.36,-122.30 --k 10000 --weight time --symmetrize min --sample_margin 0.002 --out_dir data/osm_pools/seattle_time_k10000_seed0" >&2
  exit 1
fi

EXP_ARGS=()
if [[ -n "${EXP_GAMMA}" ]]; then
  EXP_ARGS+=(--exp_gamma "${EXP_GAMMA}")
fi

python -m policy.train_matnet \
  --run_name "${RUN_NAME}" \
  --pool_dir "${POOL_DIR}" \
  --num_epochs "${NUM_EPOCHS}" \
  --num_instances "${NUM_INSTANCES}" \
  --num_val "${NUM_VAL}" \
  --num_loc "${NUM_LOC}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --log_every_n_steps "${LOG_EVERY_N_STEPS}" \
  --cost_scale "${COST_SCALE}" \
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
echo "To render a sanity GIF (road-snapped):"
echo "  python -m policy.eval_matnet --run_name ${RUN_NAME} --pbf ../../GEPA_TSP/data/osm/Seattle.osm.pbf --num_epochs ${NUM_EPOCHS} --step 5"
