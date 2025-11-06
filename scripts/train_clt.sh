#!/bin/bash

# Quick-start template for training a cross-layer transcoder.
# Edit the variables below (or wrap this script in loops) to launch runs.

set -euo pipefail

run_name="Hybrid_sampler_1e-5"
source_key="encoder_layer_0"
target_key="encoder_layer_1"
activation_epoch=300
max_samples=20000
val_ratio=0.1
num_epochs=50
batch_size=512
num_workers=0          # Set >0 when OS permits multiprocessing
seed=42

# Model / sparsity
expansion_factor=4.0
k_ratio=0.05
k_override=""          # Set to an integer to override k_ratio
tied_weights=false
init_method="kaiming_uniform"

# Optimisation
lr=1e-3
weight_decay=0.0
l1_coef=1e-3
bias_decay=1e-5
clip_grad_norm=1.0
normalize_source="standard"   # Options: none | center | standard
normalize_target="standard"

# External activation storage (set blank to use defaults)
activations_dir="/mnt/sdb1/rnarad_storage/${run_name}/activations"

# Maintenance
reinit_dead=false
reinit_freq=10
save_freq=10

# W&B logging (leave mode=disabled to skip)
wandb_project="reuben_uw"
wandb_entity="rnarad"
wandb_mode="auto"      # auto | online | offline | disabled
wandb_run_name=""      # Leave blank for auto-generated name
wandb_tags="clt smoke"

cmd=(python -m clt.train_transcoder
  --run_dir "./runs/${run_name}"
  --source_key "${source_key}"
  --target_key "${target_key}"
  --activation_epoch "${activation_epoch}"
  --max_samples "${max_samples}"
  --val_ratio "${val_ratio}"
  --num_epochs "${num_epochs}"
  --batch_size "${batch_size}"
  --num_workers "${num_workers}"
  --seed "${seed}"
  --expansion_factor "${expansion_factor}"
  --k_ratio "${k_ratio}"
  --init_method "${init_method}"
  --lr "${lr}"
  --weight_decay "${weight_decay}"
  --l1_coef "${l1_coef}"
  --bias_decay "${bias_decay}"
  --clip_grad_norm "${clip_grad_norm}"
  --normalize_source "${normalize_source}"
  --normalize_target "${normalize_target}"
  --save_freq "${save_freq}"
  --wandb_project "${wandb_project}"
  --wandb_entity "${wandb_entity}"
  --wandb_mode "${wandb_mode}"
)

if [[ -n "${wandb_run_name}" ]]; then
  cmd+=(--wandb_run_name "${wandb_run_name}")
fi

if [[ -n "${wandb_tags}" ]]; then
  # Split tags on whitespace
  read -r -a tag_array <<< "${wandb_tags}"
  cmd+=(--wandb_tags "${tag_array[@]}")
fi

if [[ "${tied_weights}" == true ]]; then
  cmd+=(--tied_weights)
fi

if [[ "${reinit_dead}" == true ]]; then
  cmd+=(--reinit_dead --reinit_freq "${reinit_freq}")
fi

if [[ -n "${k_override}" ]]; then
  cmd+=(--k "${k_override}")
fi

if [[ -n "${activations_dir}" ]]; then
  if [[ -d "${activations_dir}" ]]; then
    cmd+=(--activations_dir "${activations_dir}")
  else
    echo "Warning: activation directory '${activations_dir}' not found, falling back to defaults."
  fi
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
