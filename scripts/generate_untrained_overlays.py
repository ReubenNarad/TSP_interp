#!/usr/bin/env python3
"""
Generate feature overlays for an SAE trained on epoch-0 activations, using the
untrained policy and RandomUniform sampler matching the run's config.

Writes to: <sae_dir>/feature_analysis/

Example:
  python scripts/generate_untrained_overlays.py \
    --run_path runs/Long_RandomUniform \
    --sae_dir runs/Long_RandomUniform/sae/sae_runs/<your_sae_run> \
    --viz_instances 20 --viz_features 20 --viz_top_per_feature 10
"""

from __future__ import annotations

import sys
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


# Ensure project root on path
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from distributions import RandomUniform
from policy.policy_hooked import HookedAttentionModelPolicy
from rl4co.envs import TSPEnv
from rl4co.envs.routing.tsp.generator import TSPGenerator
from sae.model.sae_model import TopKSparseAutoencoder


def load_run_config(run_path: Path) -> Dict[str, Any]:
    return json.loads((run_path / "config.json").read_text())


def build_env(config: Dict[str, Any]) -> TSPEnv:
    num_loc = int(config.get("num_loc", 100))
    sampler = RandomUniform(num_loc=num_loc)
    env = TSPEnv(generator=TSPGenerator(num_loc=num_loc, loc_sampler=sampler))
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


def main():
    import argparse
    p = argparse.ArgumentParser(description="Generate overlays with untrained policy + trained SAE")
    p.add_argument("--run_path", type=str, required=True, help="Run dir (e.g., runs/Long_RandomUniform)")
    p.add_argument("--sae_dir", type=str, default=None, help="Path to a specific SAE run dir; if omitted uses latest")
    p.add_argument("--viz_instances", type=int, default=20, help="#instances to sample for overlays")
    p.add_argument("--viz_features", type=int, default=20, help="#top features to visualize")
    p.add_argument("--viz_top_per_feature", type=int, default=10, help="#instances per feature overlay")
    p.add_argument("--device", type=str, default=None, help="cpu|cuda|mps (auto if None)")
    args = p.parse_args()

    run_path = Path(args.run_path)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_run_config(run_path)
    env = build_env(cfg)
    policy = build_untrained_policy(env, cfg, device)
    policy.eval()

    # Resolve SAE dir
    sae_dir: Path
    if args.sae_dir:
        sae_dir = Path(args.sae_dir)
    else:
        sae_runs_dir = run_path / "sae" / "sae_runs"
        runs = sorted(sae_runs_dir.glob("*"), key=lambda p: p.stat().st_mtime)
        if not runs:
            raise FileNotFoundError(f"No SAE runs under {sae_runs_dir}")
        sae_dir = runs[-1]

    # Load SAE config and determine input/latent dims
    sae_cfg = json.loads((sae_dir / "sae_config.json").read_text())
    act_dir = run_path / "sae" / "activations"
    act0 = torch.load(act_dir / "activations_epoch_0.pt")
    act_key = sae_cfg.get("activation_key", "encoder_output")
    Xdim = int(act0[act_key].shape[1])
    latent_dim = int(float(sae_cfg.get("expansion_factor", 2.0)) * Xdim)

    sae = TopKSparseAutoencoder(
        input_dim=Xdim,
        latent_dim=latent_dim,
        k_ratio=float(sae_cfg.get("k_ratio", 0.05)),
        tied_weights=bool(sae_cfg.get("tied_weights", False)),
        bias_decay=float(sae_cfg.get("bias_decay", 1e-5)),
        dict_init=str(sae_cfg.get("init_method", "uniform")),
    ).to(device)

    final_path = sae_dir / "sae_final.pt"
    if final_path.exists():
        state = torch.load(final_path, map_location=device)
        sae.load_state_dict(state)
    else:
        ckpts = sorted((sae_dir / "checkpoints").glob("sae_epoch_*.pt"))
        if not ckpts:
            raise FileNotFoundError("No SAE weights found")
        state = torch.load(ckpts[-1], map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            sae.load_state_dict(state["model_state_dict"])
        else:
            sae.load_state_dict(state)
    sae.eval()

    # Output dirs
    out_dir = sae_dir / "feature_analysis"
    overlays_dir = out_dir / "feature_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    # Sample instances
    instances = env.reset(batch_size=[int(args.viz_instances)]).to(device)
    with torch.no_grad():
        h, _ = policy.encoder(instances)  # [B, N, D]
        B, N, D = h.shape
        flat = h.reshape(-1, D)
        Z_chunks = []
        step = 65536
        for i in range(0, flat.shape[0], step):
            _, z = sae(flat[i:i+step])
            Z_chunks.append(z)
        Z = torch.cat(Z_chunks, dim=0).reshape(B, N, -1)

    feat_mean = Z.mean(dim=(0, 1))
    k = min(int(args.viz_features), Z.shape[-1])
    top_vals, top_idx = torch.topk(feat_mean, k=k)
    top_idx = top_idx.cpu().tolist()

    # Build activation index
    activation_index: Dict[str, Dict[str, float]] = {}
    inst_means = Z.mean(dim=1)  # [B, latent]

    def plot_instance(ax, coords: np.ndarray, node_acts: np.ndarray):
        vmax = node_acts.max() if node_acts.max() > 0 else 1.0
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=node_acts, cmap='viridis',
                        s=100, alpha=1.0, edgecolors='black', linewidths=0.5,
                        vmin=0.0, vmax=vmax)
        ax.set_aspect('equal')
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([]); ax.set_yticks([])
        return sc

    for feat in top_idx:
        means_f = inst_means[:, feat]
        t = min(int(args.viz_top_per_feature), B)
        _, inst_idx = torch.topk(means_f, k=t)
        inst_idx = inst_idx.cpu().tolist()

        feat_dir = out_dir / f"feature_{feat}"
        feat_dir.mkdir(parents=True, exist_ok=True)

        for i_inst in inst_idx:
            coords = instances['locs'][i_inst].detach().cpu().numpy()
            node_acts = Z[i_inst, :, feat].detach().cpu().numpy()
            fig, ax = plt.subplots(figsize=(6, 6))
            sc = plot_instance(ax, coords, node_acts)
            avg = float(node_acts.mean())
            ax.set_title(f"Feature {feat} | Inst {i_inst} | Avg {avg:.4f}")
            cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Activation")
            fig.savefig(feat_dir / f"instance_{i_inst}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

        # Overlay all top instances on a single axes with distinct markers
        markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '<', '>']
        fig, ax = plt.subplots(figsize=(8, 8))
        from matplotlib.lines import Line2D
        legend_handles = []
        # use a common vmax across instances for consistent color scale
        try:
            vmax = float(max(Z[i, :, feat].max().item() for i in inst_idx))
            if vmax <= 0:
                vmax = 1.0
        except Exception:
            vmax = 1.0
        for k_, i_inst in enumerate(inst_idx):
            coords = instances['locs'][i_inst].detach().cpu().numpy()
            node_acts = Z[i_inst, :, feat].detach().cpu().numpy()
            mk = markers[k_ % len(markers)]
            sc = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=node_acts, cmap='viridis',
                s=40, alpha=0.9,
                marker=mk, edgecolors='black', linewidths=0.3,
                vmin=0.0, vmax=vmax,
            )
            legend_handles.append(
                Line2D([0], [0], marker=mk, color='w', label=f'Instance {i_inst}',
                       markerfacecolor='#808080', markeredgecolor='black', markersize=6)
            )
        ax.set_aspect('equal')
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(handles=legend_handles, loc='upper right', fontsize=8)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f'Feature {feat} Activation')
        ax.set_title(f'Feature {feat} Overlay Across {len(inst_idx)} Instances', fontsize=12)
        fig.savefig(out_dir / 'feature_overlays' / f'feature_{feat}_overlay.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    for i in range(instances['locs'].shape[0]):
        key = f"instance_{i}"
        activation_index[key] = {str(feat): float(inst_means[i, feat].item()) for feat in top_idx}

    (out_dir / "activation_index.json").write_text(json.dumps(activation_index, indent=2))
    print(f"[viz] Wrote overlays and activation_index to: {out_dir}")


if __name__ == "__main__":
    main()
