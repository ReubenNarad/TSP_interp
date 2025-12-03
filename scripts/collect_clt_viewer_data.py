#!/usr/bin/env python
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from clt.model import CrossLayerTranscoder
from clt.utils import (
    apply_normalization,
    collect_instances_for_overlay,
    load_env_and_policy,
    load_json,
    resolve_clt_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect CLT data for the interactive viewer.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the policy run (e.g., runs/Hybrid).")
    parser.add_argument("--pair_name", type=str, required=True, help="CLT pair directory, e.g., encoder_layer_0__to__encoder_output.")
    parser.add_argument("--clt_subdir", type=str, default="latest", help="Specific CLT run folder (default: latest).")
    parser.add_argument("--num_instances", type=int, default=12, help="Number of sampled TSP instances to store.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size when rolling out the policy.")
    parser.add_argument("--stat_samples", type=int, default=200000, help="Max flattened tokens for latent stats.")
    parser.add_argument("--top_features", type=int, default=64, help="Number of globally ranked features to cache.")
    parser.add_argument("--device", type=str, default=None, help="Device override for policy inference (cpu / cuda / mps).")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional override for viewer data directory.")
    return parser.parse_args()


def build_clt_model(config: Dict, source_dim: int, target_dim: int) -> CrossLayerTranscoder:
    inferred_k_ratio = config.get("k_ratio")
    if inferred_k_ratio is None and config.get("k") and config.get("latent_dim"):
        inferred_k_ratio = max(1, config["k"]) / config["latent_dim"]

    model = CrossLayerTranscoder(
        input_dim=source_dim,
        output_dim=target_dim,
        expansion_factor=config["expansion_factor"],
        k_ratio=inferred_k_ratio if inferred_k_ratio is not None else 0.05,
        k=config.get("k"),
        tied_weights=config.get("tied_weights", False),
        init_method=config.get("init_method", "kaiming_uniform"),
    )

    checkpoint = config["run_dir"] / "clt_best.pt"
    if not checkpoint.exists():
        checkpoint = config["run_dir"] / "clt_final.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    return model


def load_activation_tensors(
    activation_dir: Path,
    epoch: int,
    source_key: str,
    target_key: str,
):
    activation_path = activation_dir / f"activations_epoch_{epoch}.pt"
    if not activation_path.exists():
        raise FileNotFoundError(f"Activation file not found: {activation_path}")

    tensors = torch.load(activation_path, map_location="cpu")
    if source_key not in tensors:
        raise KeyError(f"Source key '{source_key}' missing from {activation_path}")
    if target_key not in tensors:
        raise KeyError(f"Target key '{target_key}' missing from {activation_path}")
    return tensors[source_key].float(), tensors[target_key].float(), tensors


def stack_actions(instances: List[Dict]) -> np.ndarray:
    lengths = [inst["actions"].shape[0] if inst["actions"] is not None else 0 for inst in instances]
    max_len = max(lengths) if lengths else 0
    actions = -np.ones((len(instances), max_len), dtype=np.int64)
    for idx, inst in enumerate(instances):
        if inst["actions"] is None:
            continue
        arr = inst["actions"].numpy().astype(np.int64)
        actions[idx, : arr.shape[0]] = arr
    return actions


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    clt_run = resolve_clt_run(run_dir, args.pair_name, args.clt_subdir)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    config = load_json(clt_run / "config.json")
    norm_stats = load_json(clt_run / "normalization_stats.json")

    source, target, _ = load_activation_tensors(
        activation_dir=Path(config["activations_dir"]),
        epoch=config["activation_epoch"],
        source_key=config["source_key"],
        target_key=config["target_key"],
    )

    if args.stat_samples and source.shape[0] > args.stat_samples:
        idx = torch.randperm(source.shape[0])[: args.stat_samples]
        source = source[idx]

    source = apply_normalization(source, norm_stats["source"], config["normalize_source"])

    clt_model = build_clt_model(
        {**config, "run_dir": clt_run},
        source_dim=source.shape[1],
        target_dim=target.shape[1],
    )

    env_bundle, policy = load_env_and_policy(run_dir, device)
    instances = collect_instances_for_overlay(
        run_dir=run_dir,
        policy=policy,
        env=env_bundle["env"],
        clt_model=clt_model,
        source_key=config["source_key"],
        norm_stats=norm_stats,
        normalize_kind=config["normalize_source"],
        num_instances=args.num_instances,
        batch_size=args.batch_size,
        device=device,
    )

    if not instances:
        raise RuntimeError("No instances were collected for viewer data.")

    locs = torch.stack([inst["locs"] for inst in instances]).numpy()
    latents = torch.stack([inst["latents"] for inst in instances]).numpy()
    actions = stack_actions(instances)
    has_actions = not np.all(actions == -1)

    viewer_dir = Path(args.output_dir) if args.output_dir else clt_run / "viz" / "viewer_data"
    viewer_dir.mkdir(parents=True, exist_ok=True)

    data_path = viewer_dir / "instances.npz"
    np.savez_compressed(data_path, locs=locs, latents=latents, actions=actions)

    latents_tensor = torch.from_numpy(latents)
    flat = latents_tensor.reshape(-1, latents_tensor.shape[-1])
    mean_activation = flat.mean(dim=0)
    mean_abs_activation = flat.abs().mean(dim=0)
    nonzero_rate = (flat != 0).float().mean(dim=0)

    instance_means = latents_tensor.mean(dim=1)
    top_k = min(args.top_features, latents_tensor.shape[-1])
    global_top = torch.topk(mean_abs_activation, k=top_k).indices.tolist()

    instance_rankings = {}
    per_instance_top = min(20, latents_tensor.shape[-1])
    for idx in range(len(instances)):
        top_indices = torch.topk(instance_means[idx], k=per_instance_top).indices.tolist()
        instance_rankings[f"instance_{idx:02d}"] = top_indices

    try:
        clt_rel_path = str(clt_run.relative_to(run_dir.parent))
    except ValueError:
        clt_rel_path = str(clt_run)

    manifest = {
        "policy_run": run_dir.name,
        "pair_name": args.pair_name,
        "clt_run": clt_run.name,
        "clt_run_path": clt_rel_path,
        "policy_run_path": str(run_dir),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "num_instances": len(instances),
        "num_nodes": int(locs.shape[1]),
        "latent_dim": int(latents.shape[2]),
        "has_actions": bool(has_actions),
        "feature_stats": {
            "mean_activation": mean_activation.tolist(),
            "mean_abs_activation": mean_abs_activation.tolist(),
            "nonzero_rate": nonzero_rate.tolist(),
            "top_features": global_top,
        },
        "instance_rankings": instance_rankings,
        "data_file": data_path.name,
        "options": {
            "num_instances": args.num_instances,
            "batch_size": args.batch_size,
            "stat_samples": args.stat_samples,
        },
    }

    with open(viewer_dir / "manifest.json", "w") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"[viewer] Saved tensors to {data_path}")
    print(f"[viewer] Saved manifest to {viewer_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
