#!/usr/bin/env python3
"""
Self-contained pipeline to:
  1) Collect epoch-0 (untrained) encoder activations for a given run config
  2) Train a Sparse Autoencoder (SAE) using an existing SAE config JSON
  3) Optionally generate simple feature overlays from the untrained policy + trained SAE

Does not modify existing codebase files; imports existing modules.

Example:
  python scripts/untrained_sae_pipeline.py \
    --run_path runs/Long_RandomUniform \
    --sae_config runs/Long_RandomUniform/sae/sae_runs/sae_l10.001_ef4.0_k0.1_04-03_10:39:46/sae_config.json \
    --num_instances 2000 \
    --do_all

Notes:
  - Activations are saved to <run_path>/sae/activations/activations_epoch_0.pt
  - SAE is trained by calling sae.train_topk.train_sae with epoch=0
  - Overlays are generated into the new SAE run dir under feature_analysis/
"""

from __future__ import annotations

import os
import sys
import json
import math
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import torch
import numpy as np
import matplotlib.pyplot as plt


# Ensure project root is importable when running from repo root
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# Imports from this repo
from distributions import RandomUniform
from policy.policy_hooked import HookedAttentionModelPolicy
from sae.train_topk import train_sae as train_sae_fn

# RL4CO env
from rl4co.envs import TSPEnv
from rl4co.envs.routing import TSPGenerator


def set_seed(seed: Optional[int]):
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_run_config(run_path: Path) -> Dict[str, Any]:
    cfg_path = run_path / "config.json"
    with open(cfg_path, "r") as f:
        return json.load(f)


def build_env_from_config(config: Dict[str, Any]) -> TSPEnv:
    num_loc = int(config.get("num_loc", 100))
    sampler = RandomUniform(num_loc=num_loc)
    generator = TSPGenerator(num_loc=num_loc, loc_sampler=sampler)
    env = TSPEnv(generator=generator)
    return env


def build_untrained_policy(env: TSPEnv, config: Dict[str, Any], device: torch.device) -> HookedAttentionModelPolicy:
    policy = HookedAttentionModelPolicy(
        env_name=env.name,
        embed_dim=int(config.get("embed_dim", 256)),
        num_encoder_layers=int(config.get("n_encoder_layers", 5)),
        num_heads=int(config.get("num_heads", 8)),
        temperature=float(config.get("temperature", 1.0)),
        dropout=float(config.get("dropout", 0.0)),
        attention_dropout=float(config.get("attention_dropout", 0.0)),
    )
    return policy.to(device)


def collect_epoch0_activations(
    run_path: Path,
    num_instances: int,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    overwrite: bool = False,
) -> Tuple[Path, Dict[str, Any]]:
    """Generate activations for an untrained (random-init) policy and save to epoch 0.

    Returns: (activation_pt_path, metadata)
    """
    set_seed(seed)

    # Resolve device
    if device is None:
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    config = load_run_config(run_path)
    env = build_env_from_config(config)

    # Generate fresh instances (not val_td)
    print(f"[collect] Generating {num_instances} instances (num_loc={config.get('num_loc', 100)})")
    data = env.reset(batch_size=[int(num_instances)]).to(device_t)

    # Build random-initialized policy
    policy = build_untrained_policy(env, config, device_t)
    policy.eval()

    # Compute encoder output directly (avoid hook reliance)
    with torch.no_grad():
        # For RL4CO AM encoder, returns (h, init_h)
        h, _ = policy.encoder(data)
        if h.dim() != 3:
            raise RuntimeError(f"Unexpected encoder output shape: {tuple(h.shape)}")
        bsz, num_nodes, feat_dim = h.shape
        flat = h.reshape(-1, feat_dim).cpu()

    # Save to run_path/sae/activations/activations_epoch_0.pt
    act_dir = run_path / "sae" / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)
    act_pt = act_dir / "activations_epoch_0.pt"
    if act_pt.exists() and not overwrite:
        print(f"[collect] Activation file exists, skipping save: {act_pt}")
    else:
        torch.save({"encoder_output": flat}, act_pt)
        print(f"[collect] Saved activations: {act_pt}")

    metadata = {
        "epoch": 0,
        "shapes": {"encoder_output": list(flat.shape)},
        "model_config": config,
        "num_instances": int(num_instances),
        "num_nodes": int(num_nodes),
    }
    meta_path = act_dir / "metadata_epoch_0.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[collect] Saved metadata: {meta_path}")

    return act_pt, metadata


@dataclass
class SAETrainArgs:
    # Match sae.train_topk expected args
    run_dir: str
    epoch: Optional[int] = 0
    activation_key: str = "encoder_output"
    normalize: bool = False
    expansion_factor: float = 2.0
    k_ratio: float = 0.05
    tied_weights: bool = False
    init_method: str = "uniform"
    batch_size: int = 512
    num_epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    l1_coef: float = 1e-3
    bias_decay: float = 1e-5
    clip_grad_norm: float = 1.0
    reinit_dead: bool = False
    reinit_freq: int = 2
    save_freq: int = 10
    num_workers: int = 4


def load_sae_config(sae_config_path: Path, run_dir: Path) -> SAETrainArgs:
    with open(sae_config_path, "r") as f:
        cfg = json.load(f)

    # Build args, defaulting missing fields
    args = SAETrainArgs(
        run_dir=str(run_dir),
        epoch=0,  # force training on epoch 0 activations
        activation_key=str(cfg.get("activation_key", "encoder_output")),
        normalize=bool(cfg.get("normalize", False)),
        expansion_factor=float(cfg.get("expansion_factor", 2.0)),
        k_ratio=float(cfg.get("k_ratio", 0.05)),
        tied_weights=bool(cfg.get("tied_weights", False)),
        init_method=str(cfg.get("init_method", "uniform")),
        batch_size=int(cfg.get("batch_size", 512)),
        num_epochs=int(cfg.get("num_epochs", 100)),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        l1_coef=float(cfg.get("l1_coef", 1e-3)),
        bias_decay=float(cfg.get("bias_decay", 1e-5)),
        clip_grad_norm=float(cfg.get("clip_grad_norm", 1.0)),
        reinit_dead=bool(cfg.get("reinit_dead", False)),
        reinit_freq=int(cfg.get("reinit_freq", 2)),
        save_freq=int(cfg.get("save_freq", 10)),
        num_workers=int(cfg.get("num_workers", 4)),
    )
    return args


def train_sae_on_epoch0(run_path: Path, sae_config_path: Path) -> Path:
    args = load_sae_config(sae_config_path, run_path)
    print("[sae] Training SAE with config:")
    print(json.dumps({k: getattr(args, k) for k in args.__dict__.keys()}, indent=2))

    # Call existing train function; returns the SAE run directory
    sae_dir = train_sae_fn(args)
    print(f"[sae] SAE training complete: {sae_dir}")
    return Path(sae_dir)


def generate_feature_overlays(
    env: TSPEnv,
    policy: HookedAttentionModelPolicy,
    sae_dir: Path,
    num_instances: int = 20,
    top_k_features: int = 20,
    instances_per_feature: int = 10,
    device: Optional[str] = None,
):
    """Create simple overlays under <sae_dir>/feature_analysis/.
    Uses untrained policy encoder + trained SAE to compute feature activations.
    """
    if device is None:
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    # Load trained SAE
    with open(sae_dir / "sae_config.json", "r") as f:
        sae_cfg = json.load(f)

    # Determine activation input dim by loading epoch 0 activations
    act_dir = sae_dir.parents[1] / "activations"  # runs/<run>/sae/activations
    act0 = torch.load(act_dir / "activations_epoch_0.pt")
    Xdim = int(act0[sae_cfg.get("activation_key", "encoder_output")].shape[1])

    # Build SAE model mirror
    from sae.model.sae_model import TopKSparseAutoencoder

    latent_dim = int(float(sae_cfg.get("expansion_factor", 2.0)) * Xdim)
    sae = TopKSparseAutoencoder(
        input_dim=Xdim,
        latent_dim=latent_dim,
        k_ratio=float(sae_cfg.get("k_ratio", 0.05)),
        tied_weights=bool(sae_cfg.get("tied_weights", False)),
        bias_decay=float(sae_cfg.get("bias_decay", 1e-5)),
        dict_init=str(sae_cfg.get("init_method", "uniform")),
    )

    # Load weights (sae_final.pt or latest checkpoint)
    final_path = sae_dir / "sae_final.pt"
    if final_path.exists():
        state = torch.load(final_path, map_location=device_t)
        sae.load_state_dict(state)
    else:
        ckpts = sorted((sae_dir / "checkpoints").glob("sae_epoch_*.pt"))
        if not ckpts:
            raise FileNotFoundError("No SAE weights found")
        state = torch.load(ckpts[-1], map_location=device_t)
        if isinstance(state, dict) and "model_state_dict" in state:
            sae.load_state_dict(state["model_state_dict"])
        else:
            sae.load_state_dict(state)
    sae = sae.to(device_t)
    sae.eval()

    # Prepare output dirs
    out_dir = sae_dir / "feature_analysis"
    overlays_dir = out_dir / "feature_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    # Generate instances and compute activations
    instances = env.reset(batch_size=[int(num_instances)]).to(device_t)
    with torch.no_grad():
        h, _ = policy.encoder(instances)
        B, N, D = h.shape
        Z = []
        # process in chunks to keep memory moderate
        flat = h.reshape(-1, D)
        step = 65536
        for i in range(0, flat.shape[0], step):
            _, z = sae(flat[i:i+step])
            Z.append(z)
        Z = torch.cat(Z, dim=0)
        Z = Z.reshape(B, N, -1)

    # Compute average activation per feature
    feat_mean = Z.mean(dim=(0, 1))  # [latent_dim]
    top_vals, top_idx = torch.topk(feat_mean, k=min(top_k_features, Z.shape[-1]))
    top_idx = top_idx.cpu().tolist()

    # Build activation index
    activation_index: Dict[str, Dict[str, float]] = {}

    # Helper to plot one instance for one feature
    def plot_instance(ax, coords: np.ndarray, node_acts: np.ndarray):
        vmax = node_acts.max() if node_acts.max() > 0 else 1.0
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=node_acts,
            cmap='viridis', s=100, alpha=1.0,
            edgecolors='black', linewidths=0.5,
            vmin=0.0, vmax=vmax,
        )
        ax.set_aspect('equal')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([]); ax.set_yticks([])
        return sc

    # Precompute instance means per feature to rank instances
    inst_means = Z.mean(dim=1)  # [B, latent_dim]

    for feat in top_idx:
        # Rank instances for this feature
        means_f = inst_means[:, feat]  # [B]
        top_inst_vals, top_inst_idx = torch.topk(means_f, k=min(instances_per_feature, B))
        top_inst_idx = top_inst_idx.cpu().tolist()

        # Save per-instance images
        feat_dir = out_dir / f"feature_{feat}"
        feat_dir.mkdir(parents=True, exist_ok=True)

        for rank, i_inst in enumerate(top_inst_idx):
            coords = instances['locs'][i_inst].detach().cpu().numpy()
            node_acts = Z[i_inst, :, feat].detach().cpu().numpy()

            fig, ax = plt.subplots(figsize=(6, 6))
            sc = plot_instance(ax, coords, node_acts)
            avg_act = float(node_acts.mean())
            ax.set_title(f"Feature {feat} | Inst {i_inst} | Avg {avg_act:.4f}")
            cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Activation")

            out_png = feat_dir / f"instance_{i_inst}.png"
            fig.savefig(out_png, dpi=150, bbox_inches='tight')
            plt.close(fig)

        # Save overlay mosaic (grid of top instances)
        cols = 5
        rows = int(math.ceil(len(top_inst_idx) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3.2, rows*3.2))
        axes = np.atleast_2d(axes)
        for k, ax in enumerate(axes.ravel()):
            if k < len(top_inst_idx):
                i_inst = top_inst_idx[k]
                coords = instances['locs'][i_inst].detach().cpu().numpy()
                node_acts = Z[i_inst, :, feat].detach().cpu().numpy()
                plot_instance(ax, coords, node_acts)
                ax.set_title(f"Inst {i_inst}", fontsize=9)
            else:
                ax.axis('off')
        fig.suptitle(f"Feature {feat} — Top {len(top_inst_idx)} Instances", fontsize=12)
        overlay_png = overlays_dir / f"feature_{feat}_overlay.png"
        fig.savefig(overlay_png, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Build activation index: for each instance, store avg activation for each top feature
    for i in range(instances['locs'].shape[0]):
        key = f"instance_{i}"
        activation_index[key] = {}
        for feat in top_idx:
            activation_index[key][str(feat)] = float(inst_means[i, feat].item())

    with open(out_dir / "activation_index.json", "w") as f:
        json.dump(activation_index, f, indent=2)
    print(f"[viz] Wrote overlays and activation_index to: {out_dir}")


def main():
    p = argparse.ArgumentParser(description="Untrained SAE pipeline (no code modifications)")
    p.add_argument("--run_path", type=str, default="runs/Long_RandomUniform", help="Run directory")
    p.add_argument("--sae_config", type=str, required=True, help="Path to SAE config JSON to mirror")
    p.add_argument("--num_instances", type=int, default=2000, help="Instances for activation collection")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--device", type=str, default=None, help="cpu|cuda|mps (auto if None)")
    p.add_argument("--overwrite_activations", action="store_true", help="Overwrite epoch-0 activations if exist")
    p.add_argument("--do_all", action="store_true", help="Run all stages: collect, train, visualize")
    p.add_argument("--collect_only", action="store_true", help="Only collect activations")
    p.add_argument("--train_only", action="store_true", help="Only train SAE from epoch-0 activations")
    p.add_argument("--visualize_only", action="store_true", help="Only generate overlays using latest trained SAE in this run")
    p.add_argument("--viz_instances", type=int, default=20, help="#instances for visualization overlays")
    p.add_argument("--viz_features", type=int, default=20, help="#top features to visualize")
    p.add_argument("--viz_top_per_feature", type=int, default=10, help="#instances per feature in overlays")
    args = p.parse_args()

    run_path = Path(args.run_path)
    sae_config_path = Path(args.sae_config)

    # Stage decisions
    do_collect = args.do_all or args.collect_only
    do_train = args.do_all or args.train_only
    do_viz = args.do_all or args.visualize_only

    if not (do_collect or do_train or do_viz):
        print("No stages selected. Use --do_all, --collect_only, --train_only, or --visualize_only")
        sys.exit(1)

    # Stage 1: Collect untrained activations (epoch 0)
    if do_collect:
        collect_epoch0_activations(
            run_path=run_path,
            num_instances=args.num_instances,
            seed=args.seed,
            device=args.device,
            overwrite=args.overwrite_activations,
        )

    # Stage 2: Train SAE on epoch-0 activations per provided config
    sae_dir: Optional[Path] = None
    if do_train:
        sae_dir = train_sae_on_epoch0(run_path, sae_config_path)

    # Stage 3: Generate overlays using the untrained policy + newly trained SAE
    if do_viz:
        # Prepare env and untrained policy
        config = load_run_config(run_path)
        env = build_env_from_config(config)
        device_t = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = build_untrained_policy(env, config, device_t)
        policy.eval()

        # Determine which SAE dir to use
        if sae_dir is None:
            # Use latest SAE run under run_path/sae/sae_runs
            sae_runs_dir = run_path / "sae" / "sae_runs"
            if not sae_runs_dir.exists():
                raise FileNotFoundError(f"No SAE runs found under {sae_runs_dir}")
            all_runs = sorted(sae_runs_dir.glob("*"), key=lambda p: p.stat().st_mtime)
            if not all_runs:
                raise FileNotFoundError(f"No SAE runs under {sae_runs_dir}")
            sae_dir = all_runs[-1]

        generate_feature_overlays(
            env=env,
            policy=policy,
            sae_dir=sae_dir,
            num_instances=args.viz_instances,
            top_k_features=args.viz_features,
            instances_per_feature=args.viz_top_per_feature,
            device=args.device,
        )


if __name__ == "__main__":
    main()
