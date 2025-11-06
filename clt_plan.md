# Cross-Layer Transcoder (CLT) Training Plan

## Objectives
- Learn sparse, overcomplete mappings that predict a downstream layer's activations from an upstream layer within the trained TSP policy.
- Reuse as much of the SAE infrastructure (data collection, run bookkeeping, visualization) as practical while isolating CLT-specific components.
- Produce CLTs that are amenable to attribution-graph style analysis (feature-to-feature edges, causal tracing hooks, sparsity guarantees).

## Assumptions & Dependencies
- We already have at least one trained policy run under `runs/<run_name>` with checkpoints and config metadata.
- Encoder activations are collected (or can be recollected) using `sae.collect_activations`. We'll extend this pipeline for CLT-specific tensors.
- Training will target encoder layers first; decoder CLTs can be added later once the pipeline is stable.
- Storage budget is sufficient for saving paired activations (expect ~2× size of single-layer SAE datasets).

## Step 1 — Pick Layer Pairs & Naming
1. Enumerate candidate source→target mappings (e.g., `encoder_layer_0 -> encoder_layer_2`, `encoder_layer_1 -> encoder_output`). Keep an initial focus on adjacent layers plus a few longer jumps inspired by Anthropic's attribution graphs.
2. Introduce a config file (e.g., `clt/configs/<run_name>.json`) or CLI args listing the layer pairs, sparsity targets, and sampling strategy.
3. Adopt a canonical naming convention for saved activations, e.g., `source_encoder_layer_0`, `target_encoder_layer_2`, and for CLT runs, e.g., `clt_L0_to_L2`.

## Step 2 — Extend Activation Collection
1. Update `EnhancedHookedPolicy` (or create `CLTHookedPolicy`) to capture:
   - Layer outputs (already available as `encoder_layer_i`).
   - Layer *inputs* when required (via hooks on `forward_pre` or by instrumenting the module's forward to stash `input[0]`).
2. Modify `sae.collect_activations` (or create `clt.collect_pairs`) to:
   - Accept `--layer_pairs` and `--num_instances_per_pair`.
   - For each pair, store matched tensors with identical token ordering: `(batch * num_nodes, hidden_dim_src)` and `(batch * num_nodes, hidden_dim_tgt)`.
   - Support an external storage root (CLI flag or `$ACTIVATIONS_ROOT`) so high-volume dumps can live on `/mnt/sdb1/rnarad_storage/`.
   - Save to `runs/<run>/clt/activations/<pair_name>/{sources.pt, targets.pt, metadata.json}` or the external mirror, recording the resolved path in `sae/activation_store.json`.
3. Record metadata per pair (layer names, shapes, number of samples, mean/std for normalization).

## Step 3 — Dataset Preparation Utilities
1. Implement `clt/datasets.py` that loads paired tensors, performs optional whitening / mean-centering, and yields minibatches `(x_src, x_tgt)`.
2. Support streaming from disk to avoid loading entire tensors for large runs (memory-mapped tensors or chunked `.pt` files).
3. Add a CLI helper `python -m clt.inspect_pairs --run_dir ...` to print stats (sparsity, norms, correlations) before training.

## Step 4 — CLT Architecture & Loss
1. Base model: sparse linear map with learned dictionary `W` and sparse codes `z`, following Anthropic’s CLT recipe:
   - Encoder: `z = topk(ReLU(W_enc x_src + b_enc))`.
   - Decoder: reconstruct approximation of target activations `ŷ = W_dec z + b_dec`.
   - Optionally tie `W_dec = W_tgt`, or allow untied weights with L2 penalties.
2. Loss components per batch:
   - Reconstruction MSE: `||ŷ - x_tgt||^2`.
   - Sparsity penalties: L1 on `z`, optional L1/L2 on dictionary rows.
   - Optional alignment loss when using normalized activations.
3. Provide hooks for orthogonality or energy constraints if later experiments need them.

## Step 5 — Training Loop, Logging & Scripts
1. Create `clt/train_transcoder.py` analogous to `sae/train_topk.py` with arguments:
   - `--run_dir`, `--pair`, `--expansion_factor`, `--k_ratio`, `--lr`, `--num_epochs`, etc.
2. Integrate W&B logging from the outset:
   - Add CLI flags/env detection for `--wandb_project`, `--wandb_run_name`, `--wandb_mode` (default to `offline` if credentials missing).
   - Log scalar metrics (losses, sparsity, cosine similarity), histograms of feature usage, and optional weight snapshots.
   - Store W&B run IDs in the CLT run metadata for easy resuming.
3. Local logging/checkpoints still mirrored for redundancy:
   - Track reconstruction loss (train/val), sparsity metrics, dead feature counts, cosine similarity between `ŷ` and `x_tgt`.
   - Save checkpoints every `save_freq` epochs under `runs/<run>/clt/clt_runs/<pair>/<timestamp>/`.
4. Provide a wrapper bash script `train_clt.sh` to iterate over layer pairs and hyperparameter grids (pass W&B args through).

## Step 6 — Evaluation & Diagnostics
1. Implement `clt/eval.py` to:
   - Compute per-feature attribution scores (e.g., how often a sparse unit activates).
   - Report explained variance / R² per target dimension.
   - Produce heatmaps of decoder weights linking source tokens/features to targets.
2. Add causal verification hooks:
   - Optionally patch policy forward pass to replace target layer activations with CLT-predicted reconstructions and measure performance drift.
   - Compare with SAE features by projecting SAE codes through CLTs to trace multi-step paths.
3. Export data for attribution graphs:
   - JSON/CSV summarizing strong edges (source feature → target head/channel) to be ingested by visualization tooling.

## Step 7 — Visualization & Integration
1. Extend `docs/` pipeline to surface CLT results (e.g., new tab for layer-to-layer edges, interactive graphs).
2. Provide notebooks or scripts demonstrating:
   - Inspecting single CLT features.
   - Combining CLT and SAE: source feature (SAE) → CLT → downstream attribution.
3. Document workflow in `README` and `context.md`, highlighting where CLT outputs live and how to reproduce analyses.

## Step 8 — Future Extensions
- Cross-run generalization: evaluate CLT trained on one policy run against checkpoints from different epochs/distributions.
- Decoder CLTs: connect encoder outputs to decoder inputs to align with attribution-graph literature.
- Structured sparsity (group lasso) to encourage token-level or head-level grouping, easing interpretability.

## Immediate Next Actions
1. Decide first batch of layer pairs and update activation collection to emit paired tensors.
2. Implement minimal CLT model + training loop, reusing SAE utilities where possible.
3. Wire up W&B logging in the new trainer (use a sandbox project, confirm offline fallback).
4. Run a small-scale experiment (e.g., 10k samples, adjacent layers) to validate pipeline end-to-end before scaling.
