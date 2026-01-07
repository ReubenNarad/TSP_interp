#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os


def _load_any(path: Path) -> Dict[str, Any]:
    if path.is_file():
        return torch.load(path, weights_only=False)

    dataset_pt = path / "dataset.pt"
    if dataset_pt.exists():
        return torch.load(dataset_pt, weights_only=False)

    raw_dir = path / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Expected {path} to contain dataset.pt or raw/ with shard_*.pt")

    shards = sorted(raw_dir.glob("shard_*.pt"))
    if not shards:
        raise FileNotFoundError(f"No shard_*.pt files found under {raw_dir}")

    shard_idx = int(os.environ.get("SHARD_IDX", "0"))
    if shard_idx < 0 or shard_idx >= len(shards):
        raise ValueError(f"Invalid shard_idx={shard_idx}, found {len(shards)} shards under {raw_dir}")
    return torch.load(shards[shard_idx], weights_only=False)


def _infer_bbox(coords: np.ndarray, pad: float) -> Tuple[float, float, float, float]:
    lon_min = float(coords[:, 0].min() - pad)
    lon_max = float(coords[:, 0].max() + pad)
    lat_min = float(coords[:, 1].min() - pad)
    lat_max = float(coords[:, 1].max() + pad)
    return lat_min, lat_max, lon_min, lon_max


def _maybe_draw_roads(ax, pbf: Optional[Path], bbox: Tuple[float, float, float, float]) -> None:
    if pbf is None:
        return
    try:
        from env.osm_tools_min import load_osm_network  # type: ignore
    except Exception:
        return
    try:
        nodes_df, edges_df = load_osm_network(pbf, bbox, network_type="driving")
        # Draw OSM edges as background lines (like GEPA plots), if geometry present.
        if "geometry" not in edges_df.columns:
            return
        segments = []
        try:
            from shapely.geometry import LineString, MultiLineString  # type: ignore

            for geom in edges_df["geometry"]:
                if isinstance(geom, LineString):
                    segments.append(np.asarray(geom.coords))
                elif isinstance(geom, MultiLineString):
                    for part in geom.geoms:
                        segments.append(np.asarray(part.coords))
        except Exception:
            return
        if segments:
            roads = LineCollection(segments, colors="0.82", linewidths=0.4, alpha=0.6, zorder=1)
            ax.add_collection(roads)
    except Exception:
        return


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot a single what-if instance colored by per-node delta score.")
    p.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset dir (with dataset.pt or raw/shard_*.pt) OR a shard_*.pt path.",
    )
    p.add_argument("--instance_idx", type=int, default=0, help="Instance index within the loaded file/dataset.")
    p.add_argument("--target", type=str, default="length", choices=["length", "time"], help="Which delta to color by.")
    p.add_argument("--out", type=str, default=None, help="Output PNG path (default: next to input).")
    p.add_argument("--title", type=str, default=None, help="Optional plot title override.")
    p.add_argument("--pad", type=float, default=0.005, help="BBox padding (lon/lat degrees) when drawing roads.")
    p.add_argument("--pbf", type=str, default=None, help="Optional .osm.pbf path to draw roads behind points.")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    data_path = Path(args.data).expanduser().resolve()
    ds = _load_any(data_path)

    locs_t = ds.get("locs")
    if not torch.is_tensor(locs_t) or locs_t.ndim != 3 or locs_t.shape[2] != 2:
        raise ValueError("Expected tensor 'locs' with shape [B,n,2] (lon/lat or xy)")
    B, n, _ = locs_t.shape

    inst = int(args.instance_idx)
    if inst < 0 or inst >= B:
        raise ValueError(f"instance_idx out of range: {inst} (B={B})")

    valid_base = ds.get("valid_base")
    valid_minus = ds.get("valid_minus")
    if not torch.is_tensor(valid_base) or valid_base.shape != (B,):
        raise ValueError("Expected 'valid_base' bool tensor [B]")
    if not torch.is_tensor(valid_minus) or valid_minus.shape != (B, n):
        raise ValueError("Expected 'valid_minus' bool tensor [B,n]")

    if args.target == "length":
        delta = ds.get("delta_length_pct")
        if not torch.is_tensor(delta) or delta.shape != (B, n):
            raise ValueError("Expected 'delta_length_pct' float tensor [B,n]")
        label = "Δ optimal tour cost (%)"
    else:
        delta = ds.get("delta_time_pct")
        if not torch.is_tensor(delta) or delta.shape != (B, n):
            raise ValueError("Expected 'delta_time_pct' float tensor [B,n]")
        label = "Δ solve wall-time (%)"

    coords = locs_t[inst].detach().cpu().numpy().astype(np.float64, copy=False)
    scores = delta[inst].detach().cpu().numpy().astype(np.float64, copy=False)

    mask = bool(valid_base[inst].item())
    if mask:
        mask_nodes = valid_minus[inst].detach().cpu().numpy().astype(bool, copy=False)
        scores = np.where(mask_nodes, scores, np.nan)
    else:
        scores[:] = np.nan

    best_idx = int(np.nanargmax(scores)) if np.isfinite(scores).any() else None
    best_val = float(scores[best_idx]) if best_idx is not None else float("nan")

    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    bbox = _infer_bbox(coords, pad=float(args.pad))
    pbf = Path(args.pbf).expanduser().resolve() if args.pbf else None
    _maybe_draw_roads(ax, pbf, bbox)

    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=scores,
        s=24,
        cmap="viridis",
        vmin=args.vmin,
        vmax=args.vmax,
        alpha=0.95,
        zorder=3,
        edgecolors="none",
    )
    if best_idx is not None:
        ax.scatter(
            [coords[best_idx, 0]],
            [coords[best_idx, 1]],
            s=90,
            facecolors="none",
            edgecolors="red",
            linewidths=2.0,
            zorder=4,
            label=f"best node (idx={best_idx}, {best_val:.3f}%)",
        )

    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.grid(True, color="0.85", linewidth=0.8, alpha=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(bbox[2], bbox[3])
    ax.set_ylim(bbox[0], bbox[1])

    title = args.title
    if title is None:
        src = data_path.name
        title = f"{src} | inst {inst} | {args.target} deltas"
    ax.set_title(title)

    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label)
    if best_idx is not None:
        ax.legend(loc="upper right")

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        if data_path.is_dir():
            out_path = data_path / f"delta_{args.target}_inst_{inst}.png"
        else:
            out_path = data_path.parent / f"delta_{args.target}_inst_{inst}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
