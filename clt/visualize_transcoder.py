import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize

from clt.model import CrossLayerTranscoder
from clt.train_transcoder import load_activation_tensors
from clt.utils import (
    apply_normalization,
    collect_instances_for_overlay,
    load_env_and_policy,
    load_json,
    resolve_clt_run,
)


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
        "--overlay_features",
        type=int,
        default=10,
        help="Number of feature overlays to generate (starting from feature 0).",
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


def visualize_feature_overlay(
    instances: List[Dict[str, np.ndarray]],
    feature_idx: int,
    output_path: Path,
) -> None:
    if not instances:
        raise ValueError("At least one instance is required to visualize overlays.")

    global_max = max(float(instance["latents"][:, feature_idx].max()) for instance in instances)
    vmax = max(global_max, 1e-6)
    norm = Normalize(vmin=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = None
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
    legend_handles: List[plt.Line2D] = []

    for idx, instance in enumerate(instances):
        locs = instance["locs"].numpy()
        activations = instance["latents"][:, feature_idx].numpy()
        marker = markers[idx % len(markers)]
        scatter = ax.scatter(
            locs[:, 0],
            locs[:, 1],
            c=activations,
            cmap="viridis",
            s=70,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.35,
            marker=marker,
            norm=norm,
            zorder=2,
        )
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="gray",
                markerfacecolor="gray",
                markersize=8,
                linewidth=0,
                label=f"Instance {idx + 1}",
            )
        )

    if scatter is None:
        raise RuntimeError("Failed to create scatter plot for overlay visualization.")

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Feature {feature_idx} activation")

    ax.set_title(f"Feature {feature_idx} overlay across {len(instances)} instances")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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

        if instances:
            latent_dim = instances[0]["latents"].shape[1]
            num_features = min(args.overlay_features, latent_dim)
            for feature_idx in range(num_features):
                out_path = overlay_dir / f"feature_{feature_idx:02d}.png"
                visualize_feature_overlay(instances, feature_idx, out_path)


if __name__ == "__main__":
    main()
