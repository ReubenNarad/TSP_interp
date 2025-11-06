#!/usr/bin/env python3
"""
Train a Sparse Autoencoder (SAE) on epoch-0 activations using an existing SAE
config JSON (mirrors hyperparameters). Calls sae.train_topk.train_sae.

The activation file expected:
  runs/<run>/sae/activations/activations_epoch_0.pt

Example:
  python scripts/train_sae_from_config.py \
    --run_path runs/Long_RandomUniform \
    --sae_config runs/Long_RandomUniform/sae/sae_runs/sae_l10.001_ef4.0_k0.1_04-03_10:39:46/sae_config.json
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# Ensure project root on path
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from sae.train_topk import train_sae as train_sae_fn


@dataclass
class SAETrainArgs:
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


def load_sae_config(sae_config_path: Path, run_dir: Path, epoch: int) -> SAETrainArgs:
    with open(sae_config_path, "r") as f:
        cfg = json.load(f)
    return SAETrainArgs(
        run_dir=str(run_dir),
        epoch=int(epoch),
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


def main():
    import argparse
    p = argparse.ArgumentParser(description="Train SAE on epoch-0 activations using existing config")
    p.add_argument("--run_path", type=str, required=True, help="Run dir (e.g., runs/Long_RandomUniform)")
    p.add_argument("--sae_config", type=str, required=True, help="Path to SAE config JSON to mirror")
    p.add_argument("--epoch", type=int, default=0, help="Activation epoch (default 0)")
    args = p.parse_args()

    run_path = Path(args.run_path)
    sae_config_path = Path(args.sae_config)

    # Build args and call existing trainer
    train_args = load_sae_config(sae_config_path, run_path, args.epoch)
    print("[sae] Training with:")
    print(json.dumps({k: getattr(train_args, k) for k in train_args.__dict__.keys()}, indent=2))

    sae_dir = train_sae_fn(train_args)
    print(f"[sae] Done. SAE run dir: {sae_dir}")


if __name__ == "__main__":
    main()

