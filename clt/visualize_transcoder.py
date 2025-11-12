import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize

from clt.model import CrossLayerTranscoder
from torchrl.data import Composite

from clt.train_transcoder import load_activation_tensors
from policy.reinforce_clipped import REINFORCEClipped
from sae.collect_activations import EnhancedHookedPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a trained cross-layer transcoder run.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the policy run (e.g., runs/Hybrid).")
    parser.add_argument(
        "--pair_name",
        type=str,
        required=True,
        help="Name of the CLT pair directory (e.g., encoder_layer_1__to__encoder_layer_4).",
    )
    parser.add_argument(
        "--clt_subdir",
        type=str,
        default="latest",
        help="Specific CLT run folder (default: follow the 'latest' symlink).",
    )
    parser.add_argument(
        "--stat_samples",
        type=int,
        default=200000,
        help="Number of flattened tokens to sample for latent stats (default: 200k).",
    )
    parser.add_argument(
        "--top_features",
        type=int,
        default=12,
        help="Number of top-average features to highlight in bar/heatmap plots.",
    )
    parser.add_argument(
        "--overlay_instances",
        type=int,
        default=4,
        help="Number of fresh TSP instances to visualize with CLT activations.",
    )
    parser.add_argument(
        "--features_per_instance",
        type=int,
        default=3,
        help="Number of strongest CLT features to render per instance.",
    )
    parser.add_argument(
        "--overlay_batch_size",
        type=int,
        default=128,
        help="Batch size to use when generating visualization instances.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override for policy inference (cpu / cuda / mps). Default: auto.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional override for where to store visual artifacts (defaults to <clt_run>/viz).",
    )
    return parser.parse_args()


def resolve_clt_run(run_dir: Path, pair_name: str, subdir: str) -> Path:
    pair_dir = run_dir / "clt" / "clt_runs" / pair_name
    if not pair_dir.exists():
        raise FileNotFoundError(f"CLT pair directory not found: {pair_dir}")

    if subdir == "latest":
        latest_link = pair_dir / "latest"
        if latest_link.is_symlink() or latest_link.exists():
            resolved = pair_dir / os.readlink(latest_link) if latest_link.is_symlink() else latest_link
            return resolved.resolve()
        # Fall back to lexicographically latest folder
        candidates = sorted([p for p in pair_dir.iterdir() if p.is_dir()])
        if not candidates:
            raise FileNotFoundError(f"No CLT runs found under {pair_dir}")
        return candidates[-1]

    resolved = pair_dir / subdir
    if not resolved.exists():
        raise FileNotFoundError(f"CLT run directory not found: {resolved}")
    return resolved


def load_json(path: Path) -> Dict:
    with open(path, "r") as fp:
        return json.load(fp)


def patch_env_specs(env) -> None:
    def _patch(spec):
        if isinstance(spec, Composite):
            if not hasattr(spec, "data_cls"):
                spec.data_cls = None
            if not hasattr(spec, "step_mdp_static"):
                spec.step_mdp_static = False
            for child in spec.values():
                if child is not None:
                    _patch(child)

    for spec_name in ["input_spec", "output_spec", "observation_spec", "reward_spec"]:
        spec = getattr(env, spec_name, None)
        if spec is not None:
            _patch(spec)


def load_env_and_policy(run_dir: Path, device: torch.device) -> Tuple[Dict, EnhancedHookedPolicy]:
    env_path = run_dir / "env.pkl"
    config_path = run_dir / "config.json"

    if not env_path.exists():
        raise FileNotFoundError(f"Environment pickle missing: {env_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Policy config missing: {config_path}")

    with open(env_path, "rb") as fp:
        env = pickle.load(fp)

    with open(config_path, "r") as fp:
        config = json.load(fp)

    patch_env_specs(env)

    checkpoint_dir = run_dir / "checkpoints"
    candidates = list(checkpoint_dir.glob("checkpoint_epoch_*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    latest = max(
        candidates,
        key=lambda p: int(p.stem.split("checkpoint_epoch_")[1]),
    )

    policy = EnhancedHookedPolicy(
        env_name=env.name,
        embed_dim=config["embed_dim"],
        num_encoder_layers=config["n_encoder_layers"],
        num_heads=8,
        temperature=config["temperature"],
        dropout=config.get("dropout", 0.0),
        attention_dropout=config.get("attention_dropout", 0.0),
    )

    model = REINFORCEClipped.load_from_checkpoint(
        latest,
        env=env,
        policy=policy,
        strict=False,
    )
    policy = model.policy.to(device)
    policy.eval()
    return {"env": env, "config": config}, policy


def apply_normalization(
    tensor: torch.Tensor,
    stats: Dict[str, Optional[Sequence[float]]],
    kind: str,
) -> torch.Tensor:
    if kind == "none" or stats is None:
        return tensor

    mean = stats.get("mean")
    std = stats.get("std")

    if kind == "center":
        if mean is None:
            return tensor
        mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
        return tensor - mean_t
    if kind == "standard":
        if mean is None or std is None:
            return tensor
        mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
        std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).clamp_min(1e-6)
        return (tensor - mean_t) / std_t
    raise ValueError(f"Unknown normalization kind '{kind}'")


def plot_histogram(latents: torch.Tensor, output_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    values = latents.flatten().detach().cpu().numpy()
    plt.hist(values, bins=100, color="#4472C4")
    plt.title("CLT Latent Activation Distribution")
    plt.xlabel("Activation value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_top_feature_usage(latents: torch.Tensor, top_k: int, output_path: Path) -> List[int]:
    mean_act = latents.mean(dim=0)
    top_values, top_indices = torch.topk(mean_act, k=min(top_k, mean_act.shape[0]))
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(top_indices)), top_values.cpu().numpy(), color="#ED7D31")
    plt.xticks(np.arange(len(top_indices)), [str(i.item()) for i in top_indices], rotation=45)
    plt.ylabel("Avg activation")
    plt.title("Top CLT Features by Mean Activation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return [idx.item() for idx in top_indices]


def plot_decoder_heatmap(
    decoder_weights: torch.Tensor,
    feature_indices: List[int],
    output_path: Path,
) -> None:
    max_idx = max(feature_indices) if feature_indices else 0
    if decoder_weights.shape[0] > max_idx:
        selected = decoder_weights[feature_indices]
    elif decoder_weights.shape[1] > max_idx:
        selected = decoder_weights[:, feature_indices].T
    else:
        raise IndexError(
            f"Decoder weight shape {decoder_weights.shape} is incompatible with indices up to {max_idx}"
        )

    plt.figure(figsize=(12, max(2, len(feature_indices) * 0.35)))
    plt.imshow(selected.detach().cpu().numpy(), aspect="auto", cmap="coolwarm")
    plt.colorbar(label="Weight value")
    plt.yticks(range(len(feature_indices)), [str(idx) for idx in feature_indices])
    plt.xlabel("Target dimension")
    plt.ylabel("Latent feature")
    plt.title("Decoder Weights for Top Features")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def draw_tour(ax, coords: np.ndarray, actions: Optional[np.ndarray]) -> None:
    if actions is None:
        return
    if len(actions) == 0:
        return
    tour = np.concatenate((actions, [actions[0]]))
    for i in range(len(tour) - 1):
        start, end = tour[i], tour[i + 1]
        dx = coords[end, 0] - coords[start, 0]
        dy = coords[end, 1] - coords[start, 1]
        ax.arrow(
            coords[start, 0],
            coords[start, 1],
            dx,
            dy,
            head_width=0.01,
            head_length=0.02,
            fc="black",
            ec="black",
            alpha=0.45,
            length_includes_head=True,
            zorder=0,
        )


def visualize_instance_features(
    instance: Dict[str, np.ndarray],
    feature_indices: List[int],
    output_path: Path,
) -> None:
    locs = instance["locs"]
    actions = instance.get("actions")
    latents = instance["latents"]

    num_features = len(feature_indices)
    fig, axes = plt.subplots(1, num_features, figsize=(4 * num_features, 4))
    if num_features == 1:
        axes = [axes]

    for ax, feature_idx in zip(axes, feature_indices):
        values = latents[:, feature_idx]
        norm = Normalize(vmin=0, vmax=max(values.max(), 1e-6))
        scatter = ax.scatter(
            locs[:, 0],
            locs[:, 1],
            c=values,
            cmap="viridis",
            s=90,
            edgecolors="black",
            norm=norm,
        )
        draw_tour(ax, locs, actions)
        ax.set_title(f"Feature {feature_idx}")
        ax.set_aspect("equal")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def collect_instances_for_overlay(
    run_dir: Path,
    policy,
    env,
    clt_model: CrossLayerTranscoder,
    source_key: str,
    norm_stats: Dict[str, Dict],
    normalize_kind: str,
    num_instances: int,
    batch_size: int,
    device: torch.device,
) -> List[Dict[str, np.ndarray]]:
    collected: List[Dict[str, np.ndarray]] = []
    clt_model = clt_model.cpu()

    while len(collected) < num_instances:
        current_batch = min(batch_size, num_instances - len(collected))
        data = env.reset(batch_size=[current_batch]).to(device)
        with torch.no_grad():
            outputs = policy(
                data,
                decode_type="greedy",
                return_actions=True,
            )
        activ = policy.activation_cache.get(source_key)
        if activ is None:
            raise RuntimeError(f"Activation '{source_key}' not found in policy cache.")

        activ_cpu = activ.detach().cpu()
        batch, num_nodes, hidden = activ_cpu.shape
        flat = activ_cpu.reshape(-1, hidden)
        normed = apply_normalization(flat, norm_stats.get("source"), normalize_kind)
        with torch.no_grad():
            latents = clt_model.encode(normed).reshape(batch, num_nodes, -1).cpu().numpy()

        actions = outputs.get("actions")
        if actions is not None:
            actions = actions.detach().cpu().numpy()

        locs = data["locs"].detach().cpu().numpy()
        for idx in range(batch):
            collected.append(
                {
                    "locs": locs[idx],
                    "actions": actions[idx] if actions is not None else None,
                    "latents": latents[idx],
                }
            )
            if len(collected) >= num_instances:
                break

    return collected


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    clt_run = resolve_clt_run(run_dir, args.pair_name, args.clt_subdir)
    viz_dir = Path(args.output_dir) if args.output_dir else clt_run / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(clt_run / "config.json")
    norm_stats = load_json(clt_run / "normalization_stats.json")

    activation_epoch = config["activation_epoch"]
    source, target, _ = load_activation_tensors(
        activation_dir=Path(config["activations_dir"]),
        epoch=activation_epoch,
        source_key=config["source_key"],
        target_key=config["target_key"],
    )

    if args.stat_samples and source.shape[0] > args.stat_samples:
        idx = torch.randperm(source.shape[0])[: args.stat_samples]
        source = source[idx]

    source = apply_normalization(source, norm_stats["source"], config["normalize_source"])

    latent_dim = int(config["expansion_factor"] * source.shape[1])
    inferred_k_ratio = config.get("k_ratio")
    if inferred_k_ratio is None and config.get("k") and config.get("latent_dim"):
        inferred_k_ratio = max(1, config["k"]) / config["latent_dim"]

    clt_model = CrossLayerTranscoder(
        input_dim=source.shape[1],
        output_dim=target.shape[1],
        expansion_factor=config["expansion_factor"],
        k_ratio=inferred_k_ratio if inferred_k_ratio is not None else 0.05,
        k=config.get("k"),
        tied_weights=config.get("tied_weights", False),
        init_method=config.get("init_method", "kaiming_uniform"),
    )

    checkpoint = clt_run / "clt_best.pt"
    if not checkpoint.exists():
        checkpoint = clt_run / "clt_final.pt"
    clt_model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    clt_model.eval()

    with torch.no_grad():
        latents = clt_model.encode(source)

    plot_histogram(latents, viz_dir / "latent_histogram.png")
    top_features = plot_top_feature_usage(latents, args.top_features, viz_dir / "top_feature_usage.png")

    if clt_model.tied_weights:
        decoder_weights = clt_model.encoder.weight.T  # Using encoder dictionary
    else:
        decoder_weights = clt_model.decoder.weight
    plot_decoder_heatmap(decoder_weights, top_features, viz_dir / "decoder_heatmap.png")

    if args.overlay_instances > 0:
        device = (
            torch.device(args.device)
            if args.device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            num_instances=args.overlay_instances,
            batch_size=args.overlay_batch_size,
            device=device,
        )

        overlay_dir = viz_dir / "overlays"
        overlay_dir.mkdir(exist_ok=True)

        for idx, instance in enumerate(instances):
            latents_tensor = torch.from_numpy(instance["latents"])
            mean_per_feature = latents_tensor.mean(dim=0)
            num_features = min(args.features_per_instance, mean_per_feature.shape[0])
            feature_ids = torch.topk(mean_per_feature, k=num_features).indices.tolist()
            out_path = overlay_dir / f"instance_{idx:02d}.png"
            visualize_instance_features(instance, feature_ids, out_path)


if __name__ == "__main__":
    main()
