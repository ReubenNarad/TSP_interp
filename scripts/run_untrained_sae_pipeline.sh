python scripts/collect_untrained_activations.py --run_path runs/Long_RandomUniform --num_instances 2000 --seed 123
python scripts/train_sae_from_config.py --run_path runs/Long_RandomUniform --sae_config runs/Long_RandomUniform/sae/sae_runs/sae_l10.001_ef4.0_k0.1_04-03_10:39:46/sae_config.json --epoch 0
python scripts/generate_untrained_overlays.py --run_path runs/Long_RandomUniform --sae_dir runs/Long_RandomUniform/sae/sae_runs/<REPLACE_WITH_SAE_RUN_DIR> --viz_instances 20 --viz_features 20 --viz_top_per_feature 10
