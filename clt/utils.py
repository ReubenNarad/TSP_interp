import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torchrl.data import Composite

from clt.model import CrossLayerTranscoder
from policy.reinforce_clipped import REINFORCEClipped
from policy.matnet_hooked import EnhancedHookedMatNetPolicy
from sae.collect_activations import EnhancedHookedPolicy


def load_json(path: Path) -> Dict:
    with open(path, "r") as fp:
        return json.load(fp)


def resolve_clt_run(run_dir: Path, pair_name: str, subdir: str) -> Path:
    pair_dir = run_dir / "clt" / "clt_runs" / pair_name
    if not pair_dir.exists():
        raise FileNotFoundError(f"CLT pair directory not found: {pair_dir}")

    if subdir == "latest":
        latest_link = pair_dir / "latest"
        if latest_link.is_symlink() or latest_link.exists():
            resolved = pair_dir / os.readlink(latest_link) if latest_link.is_symlink() else latest_link
            return resolved.resolve()
        candidates = sorted([p for p in pair_dir.iterdir() if p.is_dir()])
        if not candidates:
            raise FileNotFoundError(f"No CLT runs found under {pair_dir}")
        return candidates[-1]

    resolved = pair_dir / subdir
    if not resolved.exists():
        raise FileNotFoundError(f"CLT run directory not found: {resolved}")
    return resolved


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


def _is_matnet_run(env, config: Dict) -> bool:
    if config.get("env_name") == "atsp":
        return True
    if getattr(env, "name", None) == "atsp" and (config.get("pool_dir") or config.get("tsplib_path")):
        return True
    return False


def _resolve_policy_checkpoint(
    run_dir: Path,
    *,
    checkpoint_path: Optional[Path] = None,
    checkpoint_epoch: Optional[int] = None,
) -> Path:
    """Resolve a policy checkpoint under runs/<run>/checkpoints.

    Priority:
      1) checkpoint_path (explicit)
      2) checkpoint_epoch (by convention: checkpoint_epoch_<N>.ckpt)
      3) latest checkpoint_epoch_*.ckpt
    """
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_path is not None:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {path}")
        return path

    if checkpoint_epoch is not None:
        epoch = int(checkpoint_epoch)
        if epoch <= 0:
            raise ValueError(f"checkpoint_epoch must be >= 1, got {epoch}")
        path = checkpoint_dir / f"checkpoint_epoch_{epoch}.ckpt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint for epoch {epoch} not found: {path}")
        return path

    candidates = list(checkpoint_dir.glob("checkpoint_epoch_*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return max(
        candidates,
        key=lambda p: int(p.stem.split("checkpoint_epoch_")[1]),
    )


def load_env_and_policy(
    run_dir: Path,
    device: torch.device,
    *,
    checkpoint_path: Optional[Path] = None,
    checkpoint_epoch: Optional[int] = None,
):
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

    ckpt = _resolve_policy_checkpoint(run_dir, checkpoint_path=checkpoint_path, checkpoint_epoch=checkpoint_epoch)

    if _is_matnet_run(env, config):
        policy = EnhancedHookedMatNetPolicy(
            env_name=env.name,
            embed_dim=int(config["embed_dim"]),
            num_encoder_layers=int(config["n_encoder_layers"]),
            num_heads=int(config.get("num_heads", 8)),
            normalization=str(config.get("normalization", "instance")),
            use_graph_context=bool(config.get("use_graph_context", False)),
            bias=bool(config.get("bias", False)),
            init_embedding_mode=str(config.get("init_embedding_mode", "random_onehot")),
            tanh_clipping=float(config.get("tanh_clipping", 10.0)),
            temperature=float(config.get("temperature", 1.0)),
        )
    else:
        policy = EnhancedHookedPolicy(
            env_name=env.name,
            embed_dim=config["embed_dim"],
            num_encoder_layers=config["n_encoder_layers"],
            num_heads=int(config.get("num_heads", 8)),
            temperature=config["temperature"],
            dropout=config.get("dropout", 0.0),
            attention_dropout=config.get("attention_dropout", 0.0),
        )

    model = REINFORCEClipped.load_from_checkpoint(
        ckpt,
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
) -> List[Dict[str, torch.Tensor]]:
    collected: List[Dict[str, torch.Tensor]] = []
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
            latents = clt_model.encode(normed).reshape(batch, num_nodes, -1).cpu()

        actions = outputs.get("actions")
        if actions is not None:
            actions = actions.detach().cpu()

        locs = data["locs"].detach().cpu()
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
