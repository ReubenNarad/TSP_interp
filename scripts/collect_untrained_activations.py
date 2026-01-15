#!/usr/bin/env python3
"""
Collect epoch-0 (untrained) encoder activations for a given run, without
modifying existing code. Builds env and untrained policy from the run's config.

Saves:
  - runs/<run>/sae/activations/activations_epoch_0.pt
  - runs/<run>/sae/activations/metadata_epoch_0.json

Example:
  python scripts/collect_untrained_activations.py \
    --run_path runs/Long_RandomUniform \
    --num_instances 2000 \
    --seed 123
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch


# Ensure project root on path
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from distributions import RandomUniform
from policy.policy_hooked import HookedAttentionModelPolicy
from rl4co.envs import TSPEnv
from rl4co.envs.routing.tsp.generator import TSPGenerator


def set_seed(seed: Optional[int]):
    if seed is None:
        return
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_run_config(run_path: Path) -> Dict[str, Any]:
    cfg = json.loads((run_path / "config.json").read_text())
    return cfg


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
    p = argparse.ArgumentParser(description="Collect epoch-0 encoder activations for a run")
    p.add_argument("--run_path", type=str, required=True, help="Path to run directory (e.g., runs/Long_RandomUniform)")
    p.add_argument("--num_instances", type=int, default=2000, help="#instances to generate for activations")
    p.add_argument("--device", type=str, default=None, help="cpu|cuda|mps (auto if None)")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--overwrite", action="store_true", help="Overwrite activations if they exist")
    args = p.parse_args()

    run_path = Path(args.run_path)
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_run_config(run_path)
    env = build_env(cfg)
    data = env.reset(batch_size=[int(args.num_instances)]).to(device)

    policy = build_untrained_policy(env, cfg, device)
    policy.eval()

    with torch.no_grad():
        h, _ = policy.encoder(data)  # [B, N, D]
        if h.dim() != 3:
            raise RuntimeError(f"Unexpected encoder output shape: {tuple(h.shape)}")
        B, N, D = h.shape
        flat = h.reshape(-1, D).cpu()

    act_dir = run_path / "sae" / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)
    act_pt = act_dir / "activations_epoch_0.pt"
    if act_pt.exists() and not args.overwrite:
        print(f"[collect] File exists, not overwriting: {act_pt}")
    else:
        torch.save({"encoder_output": flat}, act_pt)
        print(f"[collect] Saved activations: {act_pt}")

    meta = {
        "epoch": 0,
        "shapes": {"encoder_output": list(flat.shape)},
        "model_config": cfg,
        "num_instances": int(args.num_instances),
        "num_nodes": int(N),
    }
    meta_path = act_dir / "metadata_epoch_0.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[collect] Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
