#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="runs/Long_RandomUniform"
VENV_PY="/home/rnarad/TSP/TSP_interp/.venv/bin/python"
LOG_VIZ="$RUN_DIR/sae/generate_overlays.log"

# Ensure log directory exists
mkdir -p "$RUN_DIR/sae"

# Wait for the latest SAE run to produce a final model
LATEST_SAE_DIR=""
while true; do
  # Find newest SAE run dir if exists
  if [ -d "$RUN_DIR/sae/sae_runs" ]; then
    LATEST_SAE_DIR=$(ls -td "$RUN_DIR"/sae/sae_runs/* 2>/dev/null | head -n1 || true)
  fi

  if [ -n "${LATEST_SAE_DIR}" ] && [ -f "${LATEST_SAE_DIR}/sae_final.pt" ]; then
    break
  fi

  sleep 20
done

# Run overlay generation using the detected latest SAE dir
"$VENV_PY" scripts/generate_untrained_overlays.py \
  --run_path "$RUN_DIR" \
  --sae_dir "$LATEST_SAE_DIR" \
  --viz_instances 20 \
  --viz_features 20 \
  --viz_top_per_feature 10 \
  > "$LOG_VIZ" 2>&1

echo "Overlays generated. See: $LOG_VIZ"
