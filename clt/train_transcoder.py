import argparse
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dotenv import load_dotenv

try:
    import wandb
except ImportError:  # pragma: no cover - wandb may be optional in some environments
    wandb = None

from clt.model import CrossLayerTranscoder


@dataclass
class NormalizationStats:
    mean: Optional[torch.Tensor]
    std: Optional[torch.Tensor]

    def to_json(self) -> Dict[str, Optional[Sequence[float]]]:
        def _convert(tensor: Optional[torch.Tensor]) -> Optional[Sequence[float]]:
            if tensor is None:
                return None
            return tensor.tolist()

        return {"mean": _convert(self.mean), "std": _convert(self.std)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_activation_tensors(
    activation_dir: Path,
    epoch: int,
    source_key: str,
    target_key: str,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Tuple[int, ...]]]:
    activation_path = activation_dir / f"activations_epoch_{epoch}.pt"
    if not activation_path.exists():
        raise FileNotFoundError(f"Activation file not found: {activation_path}")

    tensors = torch.load(activation_path)
    if source_key not in tensors:
        raise KeyError(f"Source key '{source_key}' not present in {activation_path}. "
                       f"Available keys: {list(tensors.keys())}")
    if target_key not in tensors:
        raise KeyError(f"Target key '{target_key}' not present in {activation_path}. "
                       f"Available keys: {list(tensors.keys())}")

    source = tensors[source_key].float()
    target = tensors[target_key].float()

    shapes = {key: tuple(tensor.shape) for key, tensor in tensors.items()}
    return source, target, shapes


def find_latest_activation_epoch(activation_dir: Path) -> int:
    candidates = list(activation_dir.glob("activations_epoch_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No activation tensors found in {activation_dir}")
    epochs = [int(path.stem.split("_epoch_")[1]) for path in candidates]
    return max(epochs)


def resolve_activation_dir(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.activations_dir:
        return Path(args.activations_dir).expanduser().resolve(strict=False)

    store_path = run_dir / "sae" / "activation_store.json"
    if store_path.exists():
        with open(store_path, "r") as fp:
            info = json.load(fp)
        return Path(info["activations_dir"]).expanduser().resolve(strict=False)

    activations_root = os.getenv("ACTIVATIONS_ROOT")
    if activations_root:
        return Path(activations_root).expanduser().resolve(strict=False) / run_dir.name / "activations"

    return (run_dir / "sae" / "activations").resolve()


def normalise_tensor(tensor: torch.Tensor, kind: str) -> Tuple[torch.Tensor, NormalizationStats]:
    if kind == "none":
        return tensor, NormalizationStats(None, None)
    if kind == "center":
        mean = tensor.mean(dim=0, keepdim=True)
        return tensor - mean, NormalizationStats(mean.squeeze(), None)
    if kind == "standard":
        mean = tensor.mean(dim=0, keepdim=True)
        std = tensor.std(dim=0, keepdim=True).clamp_min(1e-6)
        return (tensor - mean) / std, NormalizationStats(mean.squeeze(), std.squeeze())
    raise ValueError(f"Unknown normalisation kind '{kind}'")


def maybe_limit_samples(
    source: torch.Tensor,
    target: torch.Tensor,
    max_samples: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if max_samples is None or source.shape[0] <= max_samples:
        return source, target
    idx = torch.randperm(source.shape[0])[:max_samples]
    return source[idx], target[idx]


def init_wandb(
    args: argparse.Namespace,
    run_config: Dict,
    run_name: str,
) -> Optional["wandb.sdk.wandb_run.Run"]:
    if args.wandb_mode == "disabled":
        return None

    if wandb is None:
        print("wandb is not installed. Proceeding without W&B logging.")
        return None

    load_dotenv(dotenv_path=Path(args.repo_root) / ".env", override=False)

    mode = args.wandb_mode
    if mode == "auto":
        mode = "online" if os.getenv("WANDB_API_KEY") else "offline"

    if mode == "online" and not os.getenv("WANDB_API_KEY"):
        print("WANDB_API_KEY not found; falling back to offline logging.")
        mode = "offline"

    try:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or run_name,
            config=run_config,
            tags=args.wandb_tags,
            mode=mode,
        )
        return run
    except Exception as exc:  # pragma: no cover - depends on runtime availability
        print(f"Unable to initialise wandb ({exc}). Continuing without W&B logging.")
        return None


def make_output_dirs(run_dir: Path, pair_name: str, timestamp: str) -> Path:
    clt_dir = run_dir / "clt"
    runs_dir = clt_dir / "clt_runs" / pair_name
    output_dir = runs_dir / f"clt_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Maintain a per-pair "latest" symlink for convenience
    latest_link = runs_dir / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    try:
        latest_link.symlink_to(output_dir.name)
    except (OSError, NotImplementedError):
        # Symlinks might not be available (e.g., Windows w/out admin rights)
        pass

    return output_dir


def cosine_similarity(y_hat: torch.Tensor, y_true: torch.Tensor) -> float:
    cos = F.cosine_similarity(y_hat, y_true, dim=1)
    return cos.mean().item()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    l1_coef: float,
    bias_decay: float,
    clip_grad_norm: float,
) -> Dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "recon": 0.0,
        "l1": 0.0,
        "sparsity_l0": 0.0,
        "sparsity_l1": 0.0,
    }
    num_batches = 0

    for batch in dataloader:
        src, tgt = [tensor.to(device) for tensor in batch]
        optimizer.zero_grad(set_to_none=True)

        recon, latent = model(src)
        recon_loss = F.mse_loss(recon, tgt)
        l1_loss = l1_coef * latent.abs().sum(dim=1).mean()

        if isinstance(model, CrossLayerTranscoder) and model.tied_weights:
            bias = model.decoder_bias
        else:
            bias = model.decoder.bias if hasattr(model, "decoder") else None
        bias_loss = bias_decay * (bias.pow(2).sum() if bias is not None else torch.tensor(0.0, device=device))

        total_loss = recon_loss + l1_loss + bias_loss
        total_loss.backward()

        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()

        totals["loss"] += total_loss.item()
        totals["recon"] += recon_loss.item()
        totals["l1"] += l1_loss.item()
        totals["sparsity_l0"] += CrossLayerTranscoder.l0_sparsity(latent)
        totals["sparsity_l1"] += CrossLayerTranscoder.l1_sparsity(latent)
        num_batches += 1

    return {k: v / num_batches for k, v in totals.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "recon": 0.0,
        "sparsity_l0": 0.0,
        "sparsity_l1": 0.0,
        "cosine": 0.0,
    }
    num_batches = 0

    for batch in dataloader:
        src, tgt = [tensor.to(device) for tensor in batch]
        recon, latent = model(src)
        recon_loss = F.mse_loss(recon, tgt)
        totals["loss"] += recon_loss.item()
        totals["recon"] += recon_loss.item()
        totals["sparsity_l0"] += CrossLayerTranscoder.l0_sparsity(latent)
        totals["sparsity_l1"] += CrossLayerTranscoder.l1_sparsity(latent)
        totals["cosine"] += cosine_similarity(recon, tgt)
        num_batches += 1

    return {k: v / num_batches for k, v in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a sparse cross-layer transcoder (CLT).")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the policy run directory.")
    parser.add_argument("--source_key", type=str, required=True, help="Activation key for the upstream layer.")
    parser.add_argument("--target_key", type=str, required=True, help="Activation key for the downstream layer.")
    parser.add_argument("--activation_epoch", type=int, default=None, help="Activation epoch to use (default: latest).")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on the number of samples.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--l1_coef", type=float, default=1e-3)
    parser.add_argument("--bias_decay", type=float, default=1e-5)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--expansion_factor", type=float, default=4.0)
    parser.add_argument("--k_ratio", type=float, default=0.05)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--tied_weights", action="store_true")
    parser.add_argument("--init_method", type=str, default="kaiming_uniform",
                        choices=["kaiming_uniform", "kaiming_normal", "xavier_uniform", "xavier_normal"])
    parser.add_argument("--normalize_source", type=str, default="none",
                        choices=["none", "center", "standard"],
                        help="Normalisation applied to the source activations.")
    parser.add_argument("--normalize_target", type=str, default="none",
                        choices=["none", "center", "standard"],
                        help="Normalisation applied to the target activations.")
    parser.add_argument("--reinit_dead", action="store_true", help="Periodically reinitialise dead features.")
    parser.add_argument("--reinit_freq", type=int, default=10, help="Epoch interval for dead-feature reinit.")
    parser.add_argument("--save_freq", type=int, default=10, help="Epoch interval for saving checkpoints.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu / cuda / mps).")
    parser.add_argument("--activations_dir", type=str, default=None,
                        help="Override activation directory (defaults to pointer file or $ACTIVATIONS_ROOT).")
    parser.add_argument("--wandb_project", type=str, default="tsp-clt")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="auto", choices=["auto", "online", "offline", "disabled"])
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument("--repo_root", type=str, default=str(Path(__file__).resolve().parents[1]),
                        help="Repository root (used for locating .env)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    activation_dir = resolve_activation_dir(args, run_dir)
    if not activation_dir.exists():
        raise FileNotFoundError(f"Activation directory not found: {activation_dir}")

    if args.activation_epoch is None:
        activation_epoch = find_latest_activation_epoch(activation_dir)
    else:
        activation_epoch = args.activation_epoch

    source, target, all_shapes = load_activation_tensors(
        activation_dir=activation_dir,
        epoch=activation_epoch,
        source_key=args.source_key,
        target_key=args.target_key,
    )

    source, target = maybe_limit_samples(source, target, args.max_samples)

    source, src_stats = normalise_tensor(source, args.normalize_source)
    target, tgt_stats = normalise_tensor(target, args.normalize_target)

    num_samples = source.shape[0]
    if num_samples < 2:
        raise ValueError("Need at least two samples to create train/validation splits for CLT training.")

    val_size = max(1, int(num_samples * args.val_ratio))
    if val_size >= num_samples:
        val_size = max(1, num_samples - 1)
    train_size = num_samples - val_size

    indices = torch.randperm(num_samples)
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    train_dataset = TensorDataset(source[train_idx], target[train_idx])
    val_dataset = TensorDataset(source[val_idx], target[val_idx])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    )

    model = CrossLayerTranscoder(
        input_dim=source.shape[1],
        output_dim=target.shape[1],
        expansion_factor=args.expansion_factor,
        k_ratio=args.k_ratio,
        k=args.k,
        tied_weights=args.tied_weights,
        init_method=args.init_method,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    pair_name = f"{args.source_key}__to__{args.target_key}".replace("/", "_")
    output_dir = make_output_dirs(run_dir, pair_name, timestamp)

    run_config = {
        "source_key": args.source_key,
        "target_key": args.target_key,
        "activation_epoch": activation_epoch,
        "activations_dir": str(activation_dir),
        "num_samples": num_samples,
        "train_size": train_size,
        "val_size": val_size,
        "expansion_factor": args.expansion_factor,
        "latent_dim": model.latent_dim,
        "k": model.k,
        "normalize_source": args.normalize_source,
        "normalize_target": args.normalize_target,
        "init_method": args.init_method,
        "tied_weights": args.tied_weights,
        "optimizer": "adam",
        "lr": args.lr,
        "l1_coef": args.l1_coef,
        "bias_decay": args.bias_decay,
        "clip_grad_norm": args.clip_grad_norm,
        "batch_size": args.batch_size,
        "val_ratio": args.val_ratio,
        "max_samples": args.max_samples,
        "seed": args.seed,
    }

    wandb_run = init_wandb(args, run_config, run_name=f"{pair_name}_{timestamp}")
    history = []

    best_val_loss = float("inf")
    best_checkpoint_path = None

    for epoch in range(1, args.num_epochs + 1):
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            l1_coef=args.l1_coef,
            bias_decay=args.bias_decay,
            clip_grad_norm=args.clip_grad_norm,
        )

        val_metrics = evaluate(model=model, dataloader=val_loader, device=device)

        epoch_metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_metrics)

        if wandb_run is not None:
            flattened = {f"train/{k}": v for k, v in train_metrics.items()}
            flattened.update({f"val/{k}": v for k, v in val_metrics.items()})
            wandb_run.log(flattened, step=epoch)

        bias_term = train_metrics["loss"] - train_metrics["recon"] - train_metrics["l1"]
        print(
            f"[Epoch {epoch:03d}] "
            f"Train total: {train_metrics['loss']:.4f} "
            f"(recon={train_metrics['recon']:.4f}, l1={train_metrics['l1']:.4f}, bias={bias_term:.4f}) | "
            f"Val recon: {val_metrics['loss']:.4f} | "
            f"Val cosine: {val_metrics['cosine']:.4f}"
        )

        if args.reinit_dead and epoch % args.reinit_freq == 0:
            reinitialised = model.reset_dead_neurons(init_method=args.init_method)
            if wandb_run is not None:
                wandb_run.log({"maintenance/reinitialised_neurons": reinitialised}, step=epoch)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint_path = output_dir / "clt_best.pt"
            torch.save(model.state_dict(), checkpoint_path)
            best_checkpoint_path = checkpoint_path

        if epoch % args.save_freq == 0:
            checkpoint_path = output_dir / f"clt_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )

    final_path = output_dir / "clt_final.pt"
    torch.save(model.state_dict(), final_path)

    config_path = output_dir / "config.json"
    with open(config_path, "w") as fp:
        json.dump(run_config, fp, indent=2)

    norm_stats_path = output_dir / "normalization_stats.json"
    with open(norm_stats_path, "w") as fp:
        json.dump(
            {
                "source": src_stats.to_json(),
                "target": tgt_stats.to_json(),
            },
            fp,
            indent=2,
        )

    history_path = output_dir / "metrics.jsonl"
    with open(history_path, "w") as fp:
        for record in history:
            json.dump(record, fp)
            fp.write("\n")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as fp:
        json.dump(
            {
                "best_val_loss": best_val_loss,
                "best_checkpoint": str(best_checkpoint_path.relative_to(output_dir))
                if best_checkpoint_path is not None
                else None,
                "final_checkpoint": str(final_path.name),
            },
            fp,
            indent=2,
        )

    shape_path = output_dir / "activation_shapes.json"
    with open(shape_path, "w") as fp:
        json.dump(
            {
                "source": all_shapes[args.source_key],
                "target": all_shapes[args.target_key],
            },
            fp,
            indent=2,
        )

    if wandb_run is not None:
        wandb_run.summary["best_val_loss"] = best_val_loss
        wandb_run.summary["latent_dim"] = model.latent_dim
        wandb_run.summary["k"] = model.k
        wandb_run.finish()


if __name__ == "__main__":
    main()
