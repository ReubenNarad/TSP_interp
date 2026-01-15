#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import subprocess
import tempfile
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from env.osm_tools_min import build_graph, largest_component, load_osm_network


def _infer_run_dir(run_arg: str) -> Path:
    p = Path(run_arg).expanduser()
    if p.is_dir():
        return p.resolve()
    candidate = REPO_ROOT / "runs" / run_arg
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve run dir from '{run_arg}'")


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


def _maybe_bbox_from_pool_meta(run_dir: Path, pad: float) -> Optional[Tuple[float, float, float, float]]:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return None
    pool_dir = cfg.get("pool_dir")
    if not pool_dir:
        return None
    meta_path = Path(pool_dir) / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        bbox = meta.get("bbox")
        if not bbox or len(bbox) != 4:
            return None
        lat_min_b, lat_max_b, lon_min_b, lon_max_b = [float(x) for x in bbox]
        return (lat_min_b - pad, lat_max_b + pad, lon_min_b - pad, lon_max_b + pad)
    except Exception:
        return None


def _load_val_instance_from_run(
    run_dir: Path, instance_idx: int
) -> tuple[np.ndarray, Optional[torch.Tensor], Optional[np.ndarray], float]:
    cfg = json.loads((run_dir / "config.json").read_text())
    cost_scale = float(cfg.get("cost_scale", 1.0) or 1.0)

    cost_matrix: Optional[torch.Tensor] = None
    try:
        val_td = pickle.load(open(run_dir / "val_td.pkl", "rb"))
        cost_matrix = val_td["cost_matrix"][instance_idx].detach().cpu().to(torch.float32)
    except Exception:
        # Some runs pickle `val_td.pkl` with CUDA tensors. If CUDA is unavailable
        # (e.g. plotting on a CPU-only machine), we can still plot using saved
        # lon/lat coords + baseline tours; computing deltas requires cost_matrix.
        cost_matrix = None

    coords_path = run_dir / "val_coords_lonlat.npy"
    coords = None
    if coords_path.exists():
        coords = np.load(coords_path)[instance_idx].astype(np.float64, copy=False)
    else:
        # fallback circle coords
        if cost_matrix is None:
            raise RuntimeError(
                f"Missing {coords_path} and could not load cost_matrix from {run_dir/'val_td.pkl'}. "
                "Cannot infer N for plotting."
            )
        n = int(cost_matrix.shape[0])
        theta = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
        coords = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float64)

    val_idx_path = run_dir / "val_indices.npy"
    val_indices = None
    if val_idx_path.exists():
        val_indices = np.load(val_idx_path)[instance_idx].astype(np.int64, copy=False)

    return coords, cost_matrix, val_indices, cost_scale


def _load_baseline_tour(run_dir: Path, baseline_file: str, instance_idx: int) -> Optional[np.ndarray]:
    path = run_dir / baseline_file
    if not path.exists():
        return None
    base = pickle.load(open(path, "rb"))
    try:
        tour = base["actions"][0][instance_idx].detach().cpu().numpy().astype(np.int64, copy=False)
    except Exception:
        return None
    return tour


def _int_weights(cost: np.ndarray, scale_factor: float, rounding: str) -> np.ndarray:
    mat = np.asarray(cost, dtype=np.float64)
    mat = np.nan_to_num(mat, nan=1e12, posinf=1e12, neginf=1e12)
    mat[mat < 0] = 1e12
    np.fill_diagonal(mat, 0.0)
    if rounding == "ceil":
        return np.ceil(mat * float(scale_factor) - 1e-9).astype(np.int64)
    if rounding == "rint":
        return np.rint(mat * float(scale_factor)).astype(np.int64)
    raise ValueError(f"Unknown rounding: {rounding}")


def _concorde_solve_full_matrix_int(mat_int: np.ndarray, *, timeout_sec: float) -> Optional[list[int]]:
    n = int(mat_int.shape[0])
    with tempfile.TemporaryDirectory(prefix="whatif_plot_") as tmp:
        tmp = Path(tmp)
        tsp = tmp / "inst.tsp"
        sol = tmp / "inst.sol"
        # Write minimal TSPLIB
        with tsp.open("w", encoding="utf-8") as f:
            f.write("NAME: inst\n")
            f.write("TYPE: TSP\n")
            f.write(f"DIMENSION: {n}\n")
            f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
            f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
            f.write("EDGE_WEIGHT_SECTION\n")
            for row in mat_int:
                f.write(" ".join(str(int(x)) for x in row) + "\n")
            f.write("EOF\n")

        try:
            subprocess.run(
                ["concorde", "-o", sol.name, tsp.name],
                cwd=str(tmp),
                check=True,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_sec if timeout_sec and timeout_sec > 0 else None,
            )
        except Exception:
            return None
        if not sol.exists():
            return None
        # Concorde sol is 0-based
        tour = []
        for i, line in enumerate(sol.read_text(encoding="utf-8", errors="replace").splitlines()):
            if i == 0:
                continue
            for tok in line.strip().split():
                v = int(tok)
                if v == -1:
                    break
                tour.append(v)
        if len(tour) != n:
            return None
        mn, mx = min(tour), max(tour)
        if mn == 1 and mx == n:
            tour = [t - 1 for t in tour]
        if min(tour) != 0 or max(tour) != n - 1:
            return None
        if len(set(tour)) != n:
            return None
        return tour


def _tour_cost_int(mat_int: np.ndarray, tour0: list[int]) -> int:
    idx = np.asarray(tour0, dtype=np.int64)
    nxt = np.roll(idx, -1)
    return int(mat_int[idx, nxt].sum())


def _compute_deltas_for_instance(cost_matrix: torch.Tensor, *, cost_scale: float, rounding: str, timeout_sec: float) -> np.ndarray:
    cm = cost_matrix.detach().cpu().numpy().astype(np.float64, copy=False)
    n = int(cm.shape[0])
    mat_int = _int_weights(cm, scale_factor=cost_scale, rounding=rounding)
    tour = _concorde_solve_full_matrix_int(mat_int, timeout_sec=timeout_sec)
    if tour is None:
        return np.full((n,), np.nan, dtype=np.float64)
    base_cost = _tour_cost_int(mat_int, tour)
    deltas = np.full((n,), np.nan, dtype=np.float64)
    for i in range(n):
        keep = [k for k in range(n) if k != i]
        sub = mat_int[np.ix_(keep, keep)]
        tour_m = _concorde_solve_full_matrix_int(sub, timeout_sec=timeout_sec)
        if tour_m is None:
            continue
        minus_cost = _tour_cost_int(sub, tour_m)
        # If rounding breaks monotonicity, clamp negatives for visualization.
        delta = max(0.0, float(base_cost - minus_cost))
        deltas[i] = 100.0 * delta / max(1.0, float(base_cost))
    return deltas


def _build_snap_graph(pbf: Path, bbox: Tuple[float, float, float, float]):
    nodes_df, edges_df = load_osm_network(pbf, bbox, network_type="driving")
    g_full = build_graph(edges_df=edges_df, nodes_df=nodes_df, weight="time", default_kmh=35.0, convert_mph=False)
    return largest_component(g_full), edges_df


def _map_cities_to_graph(run_dir: Path, snap_graph, instance_indices: Optional[np.ndarray], coords: np.ndarray) -> np.ndarray:
    # Prefer exact node-id mapping when we have pool indices + pool node_ids.
    if instance_indices is not None:
        cfg = json.loads((run_dir / "config.json").read_text())
        pool_dir = cfg.get("pool_dir")
        if pool_dir:
            node_ids_path = Path(pool_dir) / "node_ids.npy"
            if node_ids_path.exists():
                base_node_ids = np.load(node_ids_path)
                mapped = []
                for i in instance_indices.tolist():
                    nid = int(base_node_ids[int(i)])
                    gi = snap_graph.id_to_idx.get(nid)
                    mapped.append(int(gi) if gi is not None else -1)
                mapped = np.asarray(mapped, dtype=np.int64)
                if (mapped >= 0).all():
                    return mapped

    # Fallback: nearest neighbor in lon/lat space
    from scipy.spatial import cKDTree

    tree = cKDTree(snap_graph.coords_lonlat)
    _d, idxs = tree.query(coords)
    return idxs.astype(np.int64)


def _plot_tour_snapped(
    ax,
    coords: np.ndarray,
    tour: np.ndarray,
    snap_graph,
    city_to_graph: np.ndarray,
    *,
    zorder: float = 4,
) -> None:
    import networkit as nk  # type: ignore

    g = snap_graph.g
    coords_lonlat = snap_graph.coords_lonlat
    closed = np.concatenate([tour, tour[:1]])
    path_cache: dict[tuple[int, int], np.ndarray] = {}
    for a, b in zip(closed[:-1], closed[1:]):
        src = int(city_to_graph[int(a)])
        tgt = int(city_to_graph[int(b)])
        key = (src, tgt)
        poly = path_cache.get(key)
        if poly is None:
            d = nk.distance.Dijkstra(g, src, storePaths=True, target=tgt)
            d.run()
            path_idx = d.getPath(tgt)
            if not path_idx:
                poly = np.stack([coords[int(a)], coords[int(b)]], axis=0).astype(np.float32)
            else:
                poly = coords_lonlat[np.asarray(path_idx, dtype=int)]
            path_cache[key] = poly
        ax.plot(poly[:, 0], poly[:, 1], color="tab:red", linewidth=1.1, alpha=0.85, zorder=zorder)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot a single what-if instance colored by per-node delta score.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset dir (with dataset.pt or raw/shard_*.pt) OR a shard_*.pt path.",
    )
    g.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run dir or run name under runs/ to plot from val_td.pkl (matches policy/eval_matnet plots).",
    )
    p.add_argument("--instance_idx", type=int, default=0, help="Instance index within the loaded file/dataset.")
    p.add_argument("--target", type=str, default="length", choices=["length", "time"], help="Which delta to color by.")
    p.add_argument("--out", type=str, default=None, help="Output PNG path (default: next to input).")
    p.add_argument("--title", type=str, default=None, help="Optional plot title override.")
    p.add_argument("--pad", type=float, default=0.005, help="BBox padding (lon/lat degrees) when drawing roads.")
    p.add_argument("--pbf", type=str, default=None, help="Optional .osm.pbf path to draw roads behind points.")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--baseline_file", type=str, default="baseline_concorde_128.pkl", help="Concorde baseline file in run dir.")
    p.add_argument("--draw_tour", action="store_true", help="Overlay the (baseline) tour in red.")
    p.add_argument(
        "--minimal",
        action="store_true",
        help="Simplified aesthetic: no title/axes/grid/colorbar/legend (just map, nodes, tour).",
    )
    p.add_argument(
        "--compute_deltas",
        action="store_true",
        help="Compute deltas for the selected run val instance by re-solving base and each removal with Concorde (slow).",
    )
    p.add_argument("--concorde_timeout_sec", type=float, default=60.0)
    p.add_argument("--rounding", type=str, default="rint", choices=["rint", "ceil"], help="Edge weight integerization for deltas.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    inst = int(args.instance_idx)

    run_dir = None
    val_indices = None
    tour = None
    if args.run is not None:
        run_dir = _infer_run_dir(args.run)
        coords, cost_matrix, val_indices, cost_scale = _load_val_instance_from_run(run_dir, inst)
        if args.compute_deltas:
            if cost_matrix is None:
                raise RuntimeError(
                    "Cannot --compute_deltas because cost_matrix could not be loaded from val_td.pkl. "
                    "This can happen if val_td.pkl was pickled with CUDA tensors but CUDA is unavailable."
                )
            scores = _compute_deltas_for_instance(
                cost_matrix, cost_scale=cost_scale, rounding=str(args.rounding), timeout_sec=float(args.concorde_timeout_sec)
            )
        else:
            scores = None if args.minimal else np.full((coords.shape[0],), np.nan, dtype=np.float64)

        if args.draw_tour:
            tour = _load_baseline_tour(run_dir, args.baseline_file, inst)
        label = "Δ optimal tour cost (%)" if args.target == "length" else "Δ solve wall-time (%)"
        src = run_dir.name
    else:
        data_path = Path(args.data).expanduser().resolve()
        ds = _load_any(data_path)

        locs_t = ds.get("locs")
        if not torch.is_tensor(locs_t) or locs_t.ndim != 3 or locs_t.shape[2] != 2:
            raise ValueError("Expected tensor 'locs' with shape [B,n,2] (lon/lat or xy)")
        B, n, _ = locs_t.shape
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
        src = data_path.name

    if scores is None:
        best_idx = None
        best_val = float("nan")
    else:
        best_idx = int(np.nanargmax(scores)) if np.isfinite(scores).any() else None
        best_val = float(scores[best_idx]) if best_idx is not None else float("nan")

    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    bbox = None
    if run_dir is not None:
        bbox = _maybe_bbox_from_pool_meta(run_dir, pad=float(args.pad))
    if bbox is None:
        bbox = _infer_bbox(coords, pad=float(args.pad))
    pbf = Path(args.pbf).expanduser().resolve() if args.pbf else None
    _maybe_draw_roads(ax, pbf, bbox)

    # Optional snapped tour overlay (matches policy/eval_matnet aesthetics).
    if tour is not None and pbf is not None and run_dir is not None:
        snap_graph, _edges_df = _build_snap_graph(pbf, bbox)
        city_to_graph = _map_cities_to_graph(run_dir, snap_graph, val_indices, coords)
        _plot_tour_snapped(ax, coords, tour, snap_graph, city_to_graph, zorder=3)

    base_s = 24
    outline_extra = 1.8  # points; renders an outside-only outline without shrinking the filled marker
    outline_s = (math.sqrt(base_s) + outline_extra) ** 2

    # Black "halo" underneath, then the colored marker on top (gives an outside-only outline).
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c="black",
        s=outline_s,
        alpha=0.95,
        zorder=4.8,
        edgecolors="none",
    )

    if scores is None:
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c="tab:blue",
            s=base_s,
            alpha=0.95,
            zorder=5,
            edgecolors="none",
        )
    else:
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=scores,
            s=base_s,
            cmap="viridis",
            vmin=args.vmin,
            vmax=args.vmax,
            alpha=0.95,
            zorder=5,
            edgecolors="none",
        )
    if best_idx is not None and not args.minimal:
        ring_extra = 4.5  # additional points beyond the node outline
        ring_s = (math.sqrt(base_s) + outline_extra + ring_extra) ** 2
        ax.scatter(
            [coords[best_idx, 0]],
            [coords[best_idx, 1]],
            s=ring_s,
            facecolors="none",
            edgecolors="red",
            linewidths=2.0,
            zorder=6,
            label=f"best node (idx={best_idx}, {best_val:.3f}%)",
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(bbox[2], bbox[3])
    ax.set_ylim(bbox[0], bbox[1])

    if args.minimal:
        ax.set_axis_off()
        ax.grid(False)
    else:
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.grid(True, color="0.85", linewidth=0.8, alpha=0.8)

        title = args.title
        if title is None:
            title = f"{src} | inst {inst} | {args.target} deltas"
        ax.set_title(title)

        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(label)
        if best_idx is not None:
            ax.legend(loc="upper right")

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        if run_dir is not None:
            out_path = run_dir / f"delta_{args.target}_inst_{inst}.png"
        else:
            if data_path.is_dir():
                out_path = data_path / f"delta_{args.target}_inst_{inst}.png"
            else:
                out_path = data_path.parent / f"delta_{args.target}_inst_{inst}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.minimal:
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.0)
    else:
        fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
