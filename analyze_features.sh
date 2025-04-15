#!/bin/bash

# Use simple run names instead of full paths
run_name="Test_Clusters_8_layers"
sae_run_name="sae_l10.001_ef4.0_k0.1_04-10_11:06:06"
# Construct full paths internally in the script
run_path="./runs/${run_name}"
sae_path="${run_path}/sae/sae_runs/${sae_run_name}"

# Hyperparameters
num_instances=10
batch_size=16
num_features=40

# Run the feature analysis
python -m sae.feature_analysis \
  --run_path "${run_path}" \
  --sae_path "${sae_path}" \
  --num_instances ${num_instances} \
  --batch_size ${batch_size} \
  --num_features ${num_features} \
  --show_solution \
  --no_multi_feature \
  # --html_only

echo "Feature analysis complete!"