#!/bin/bash
set -euo pipefail

# Tiny smoke run (CPU-friendly)
RUN_NAME="${RUN_NAME:-matnet_seattle_smoke}"
TSPLIB_PATH="${TSPLIB_PATH:-../../GEPA_TSP/data/eval/structured_seattle_time_400_seed20251125.tsp}"

python -m policy.train_matnet \
  --run_name "$RUN_NAME" \
  --num_epochs 2 \
  --num_instances 512 \
  --num_val 64 \
  --num_loc 30 \
  --batch_size 64 \
  --lr 1e-4 \
  --checkpoint_freq 1 \
  --embed_dim 128 \
  --n_encoder_layers 2 \
  --num_heads 8 \
  --normalization instance \
  --tsplib_path "$TSPLIB_PATH" \
  --symmetrize none \
  --seed 0

