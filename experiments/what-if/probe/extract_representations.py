#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _add_repo_to_path() -> None:
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))


def _as_device(device_str: Optional[str]) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract per-node policy representations aligned to a what-if dataset.",
    )
    p.add_argument("--data_dir", type=str, required=True, help="Directory containing merged dataset.pt")
    p.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Policy run dir. If omitted, uses dataset meta['run_dir'] when present.",
    )
    p.add_argument(
        "--activation_key",
        type=str,
        default="encoder_output",
        help="Activation key to extract (e.g., encoder_output, encoder_layer_0). Ignored if --activation_keys is set.",
    )
    p.add_argument(
        "--activation_keys",
        type=str,
        default=None,
        help="Comma-separated list of activation keys to concatenate along feature dim (overrides --activation_key).",
    )
    p.add_argument("--batch_size", type=int, default=32, help="Number of instances per forward pass.")
    p.add_argument("--device", type=str, default=None, help="Device string, e.g. cuda, cpu.")
    p.add_argument("--out_path", type=str, default=None, help="Output .pt path (default: <data_dir>/probe_reps.pt)")
    p.add_argument(
        "--matnet_init_embedding_mode",
        type=str,
        default=None,
        choices=["random_onehot", "random", "onehot"],
        help="Override MatNet init embedding mode during extraction (useful to remove injected randomness).",
    )

    p.add_argument("--compute_sae", action="store_true", help="Also compute SAE latents for each node.")
    p.add_argument(
        "--sae_dir",
        type=str,
        default=None,
        help="SAE directory containing sae_final.pt + sae_config.json (default: <run_dir>/sae if present).",
    )
    p.add_argument(
        "--sae_dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Storage dtype for X_sae (float16 saves space).",
    )
    return p


def _load_dataset(data_dir: Path) -> Dict:
    dataset_path = data_dir / "dataset.pt"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset.pt at {dataset_path}")
    return torch.load(dataset_path, weights_only=False)


def _resolve_run_dir(ds: Dict, run_dir_arg: Optional[str]) -> Path:
    if run_dir_arg:
        run_dir = Path(run_dir_arg).expanduser().resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"run_dir not found: {run_dir}")
        return run_dir

    meta = ds.get("meta", {})
    run_dir_from_meta = meta.get("run_dir") if isinstance(meta, dict) else None
    if not run_dir_from_meta:
        raise ValueError("--run_dir not provided and dataset meta['run_dir'] missing")
    run_dir = Path(run_dir_from_meta).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir from dataset meta not found: {run_dir}")
    return run_dir


def _resolve_sae_dir(run_dir: Path, sae_dir_arg: Optional[str]) -> Path:
    if sae_dir_arg:
        sae_dir = Path(sae_dir_arg).expanduser().resolve()
        if not sae_dir.exists():
            raise FileNotFoundError(f"sae_dir not found: {sae_dir}")
        return sae_dir

    candidate = run_dir / "sae"
    if (candidate / "sae_final.pt").exists() and (candidate / "sae_config.json").exists():
        return candidate

    latest = run_dir / "sae" / "latest"
    if latest.exists():
        resolved = latest.resolve()
        if (resolved / "sae_final.pt").exists() and (resolved / "sae_config.json").exists():
            return resolved

    raise FileNotFoundError(
        "Could not infer sae_dir. Provide --sae_dir pointing to a folder with sae_final.pt + sae_config.json."
    )


def _load_sae(sae_dir: Path, device: torch.device):
    _add_repo_to_path()
    from sae.model.sae_model import TopKSparseAutoencoder

    cfg_path = sae_dir / "sae_config.json"
    weights_path = sae_dir / "sae_final.pt"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing sae_config.json at {cfg_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing sae_final.pt at {weights_path}")

    with open(cfg_path, "r") as fp:
        cfg = json.load(fp)

    state_dict = torch.load(weights_path, weights_only=False)
    if not isinstance(state_dict, dict) or "encoder.weight" not in state_dict:
        raise ValueError(f"Unexpected SAE state dict at {weights_path}")

    latent_dim, input_dim = state_dict["encoder.weight"].shape
    k_ratio = float(cfg.get("k_ratio", 0.05))

    tied_weights = bool(cfg.get("tied_weights", False))
    model = TopKSparseAutoencoder(
        input_dim=int(input_dim),
        latent_dim=int(latent_dim),
        k_ratio=k_ratio,
        tied_weights=tied_weights,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, cfg


def _activation_to_tensor(activation, key: str) -> torch.Tensor:
    if torch.is_tensor(activation):
        return activation
    if isinstance(activation, tuple):
        if len(activation) == 2 and torch.is_tensor(activation[0]):
            return activation[0]
        raise TypeError(f"Activation '{key}' is a tuple of unsupported shape/types: {type(activation)} len={len(activation)}")
    raise TypeError(f"Activation '{key}' has unsupported type: {type(activation)}")


def main() -> None:
    os.environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")
    args = build_arg_parser().parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    ds = _load_dataset(data_dir)

    locs = ds.get("locs")
    if not torch.is_tensor(locs) or locs.ndim != 3 or locs.shape[2] != 2:
        raise ValueError("dataset.pt missing 'locs' tensor with shape [B,n,2]")

    cost_matrix = ds.get("cost_matrix")
    if cost_matrix is not None:
        if not torch.is_tensor(cost_matrix) or cost_matrix.ndim != 3:
            raise ValueError("dataset.pt has 'cost_matrix' but it's not a tensor with shape [B,n,n]")
        if cost_matrix.shape[0] != locs.shape[0] or cost_matrix.shape[1] != locs.shape[1] or cost_matrix.shape[2] != locs.shape[1]:
            raise ValueError(
                f"dataset.pt has inconsistent shapes: locs={tuple(locs.shape)} cost_matrix={tuple(cost_matrix.shape)}"
            )

    valid_base = ds.get("valid_base")
    valid_minus = ds.get("valid_minus")
    delta_length_pct = ds.get("delta_length_pct")
    delta_time_pct = ds.get("delta_time_pct")
    if not torch.is_tensor(valid_base) or valid_base.shape != (locs.shape[0],):
        raise ValueError("dataset.pt missing 'valid_base' bool tensor [B]")
    if not torch.is_tensor(valid_minus) or valid_minus.shape != (locs.shape[0], locs.shape[1]):
        raise ValueError("dataset.pt missing 'valid_minus' bool tensor [B,n]")
    if not torch.is_tensor(delta_length_pct) or delta_length_pct.shape != (locs.shape[0], locs.shape[1]):
        raise ValueError("dataset.pt missing 'delta_length_pct' float tensor [B,n]")
    if not torch.is_tensor(delta_time_pct) or delta_time_pct.shape != (locs.shape[0], locs.shape[1]):
        raise ValueError("dataset.pt missing 'delta_time_pct' float tensor [B,n]")

    run_dir = _resolve_run_dir(ds, args.run_dir)
    device = _as_device(args.device)

    _add_repo_to_path()
    from clt.utils import load_env_and_policy

    info, policy = load_env_and_policy(run_dir=run_dir, device=device)
    env = info["env"]

    if args.matnet_init_embedding_mode is not None:
        try:
            from policy.matnet_custom import DeterministicOneHotMatNetInitEmbedding
            from rl4co.models.nn.env_embeddings.init import MatNetInitEmbedding
        except Exception as e:
            raise RuntimeError(f"Failed to import MatNet init embedding utilities: {e}") from e

        if getattr(env, "name", None) != "atsp":
            raise ValueError("--matnet_init_embedding_mode is only valid for MatNet/ATSP runs")

        mode = str(args.matnet_init_embedding_mode)
        if not hasattr(policy, "encoder") or not hasattr(policy.encoder, "init_embedding"):
            raise ValueError("Loaded policy does not appear to be MatNet (missing encoder.init_embedding)")

        embed_dim = int(info["config"].get("embed_dim", 256))
        if mode == "onehot":
            policy.encoder.init_embedding = DeterministicOneHotMatNetInitEmbedding(embed_dim=embed_dim).to(device)
        elif mode == "random":
            policy.encoder.init_embedding = MatNetInitEmbedding(embed_dim=embed_dim, mode="Random").to(device)
        elif mode == "random_onehot":
            policy.encoder.init_embedding = MatNetInitEmbedding(embed_dim=embed_dim, mode="RandomOneHot").to(device)
        else:
            raise ValueError(f"Unknown --matnet_init_embedding_mode: {mode}")

    activation_keys = None
    if args.activation_keys:
        activation_keys = [k.strip() for k in str(args.activation_keys).split(",") if k.strip()]
    if not activation_keys:
        activation_keys = [str(args.activation_key)]

    sae = None
    sae_cfg = None
    sae_dir = None
    sae_dtype = torch.float32 if args.sae_dtype == "float32" else torch.float16
    if args.compute_sae:
        sae_dir = _resolve_sae_dir(run_dir, args.sae_dir)
        sae, sae_cfg = _load_sae(sae_dir, device=device)

    B, n, _ = locs.shape
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("--batch_size must be >= 1")

    X_parts = []
    X_sae_parts = []
    y_parts = []
    valid_parts = []
    inst_parts = []
    node_parts = []

    with torch.inference_mode():
        offset = 0
        while offset < B:
            end = min(B, offset + batch_size)
            current = end - offset

            locs_batch = locs[offset:end].to(torch.float32)

            if cost_matrix is not None:
                from tensordict import TensorDict

                cm_batch = cost_matrix[offset:end].to(torch.float32).to(device)
                td = env.reset(td=TensorDict({"cost_matrix": cm_batch}, batch_size=[int(current)])).to(device)
            else:
                td = env.reset(batch_size=[current]).to(device)

            if "locs" in td.keys(True, True):
                td["locs"] = locs_batch.to(device)

            if hasattr(policy, "clear_cache"):
                policy.clear_cache()
            _ = policy.encoder(td)

            act_parts = []
            for key in activation_keys:
                activation = policy.activation_cache.get(key)
                if activation is None:
                    available = sorted(list(policy.activation_cache.keys()))
                    raise KeyError(f"Activation '{key}' not found. Available: {available}")
                activation_t = _activation_to_tensor(activation, key)
                if activation_t.ndim != 3 or activation_t.shape[0] != current or activation_t.shape[1] != n:
                    raise ValueError(
                        f"Activation '{key}' expected [B,n,d] = [{current},{n},d], got {tuple(activation_t.shape)}"
                    )
                act_parts.append(activation_t)

            activation_cat = torch.cat(act_parts, dim=-1) if len(act_parts) > 1 else act_parts[0]

            flat = activation_cat.reshape(-1, activation_cat.shape[-1]).detach().to("cpu")
            X_parts.append(flat)

            pair_valid = valid_base[offset:end].unsqueeze(1) & valid_minus[offset:end]
            y = torch.stack(
                [
                    delta_length_pct[offset:end].reshape(-1),
                    delta_time_pct[offset:end].reshape(-1),
                ],
                dim=1,
            ).to(torch.float32)
            y_parts.append(y)
            valid_parts.append(pair_valid.reshape(-1).to(torch.bool))

            inst_ids = torch.arange(offset, end, dtype=torch.int64).repeat_interleave(n)
            node_ids = torch.arange(n, dtype=torch.int64).repeat(current)
            inst_parts.append(inst_ids)
            node_parts.append(node_ids)

            if sae is not None:
                z_parts = []
                for key, act in zip(activation_keys, act_parts):
                    if int(act.shape[-1]) != int(sae.input_dim):
                        raise ValueError(
                            f"SAE input_dim mismatch for activation '{key}': SAE expects {sae.input_dim}, got {int(act.shape[-1])}"
                        )
                    z_parts.append(sae.encode(act.reshape(-1, act.shape[-1])))
                z = torch.cat(z_parts, dim=1) if len(z_parts) > 1 else z_parts[0]
                X_sae_parts.append(z.to("cpu", dtype=sae_dtype))

            offset = end

    out = {
        "meta": {
            "run_dir": str(run_dir),
            "data_dir": str(data_dir),
            "activation_key": str(args.activation_key),
            "activation_keys": activation_keys,
            "num_instances": int(B),
            "num_loc": int(n),
            "repr_dim": int(X_parts[0].shape[1]) if X_parts else 0,
            "device_used": str(device),
            "compute_sae": bool(args.compute_sae),
            "sae_dir": str(sae_dir) if sae_dir is not None else None,
            "sae_cfg": sae_cfg,
            "sae_dtype": str(args.sae_dtype),
            "matnet_init_embedding_mode": str(args.matnet_init_embedding_mode) if args.matnet_init_embedding_mode else None,
            "label_names": ["delta_length_pct", "delta_time_pct"],
        },
        "y_names": ["delta_length_pct", "delta_time_pct"],
        "instance_id": torch.cat(inst_parts, dim=0) if inst_parts else torch.empty((0,), dtype=torch.int64),
        "node_id": torch.cat(node_parts, dim=0) if node_parts else torch.empty((0,), dtype=torch.int64),
        "valid": torch.cat(valid_parts, dim=0) if valid_parts else torch.empty((0,), dtype=torch.bool),
        "y": torch.cat(y_parts, dim=0) if y_parts else torch.empty((0, 2), dtype=torch.float32),
        "X_resid": torch.cat(X_parts, dim=0) if X_parts else torch.empty((0, 0), dtype=torch.float32),
    }
    if X_sae_parts:
        out["X_sae"] = torch.cat(X_sae_parts, dim=0)

    out_path = Path(args.out_path).expanduser().resolve() if args.out_path else (data_dir / "probe_reps.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"[extract] wrote: {out_path}")


if __name__ == "__main__":
    main()
