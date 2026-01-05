from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import numpy.lib.format

from .osm_tools import (
    BBox,
    Weight,
    all_pairs_dijkstra_subset,
    build_graph,
    largest_component,
    load_osm_network,
    sample_nodes_in_bbox,
)
from .pool import save_pool


def _parse_bbox(s: str) -> BBox:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have 4 comma-separated floats: lat_min,lat_max,lon_min,lon_max")
    return (parts[0], parts[1], parts[2], parts[3])


def _shrink_bbox(bbox: BBox, margin: float) -> BBox:
    lat_min, lat_max, lon_min, lon_max = bbox
    out = (lat_min + margin, lat_max - margin, lon_min + margin, lon_max - margin)
    if out[0] >= out[1] or out[2] >= out[3]:
        raise ValueError(f"sample_margin={margin} is too large for bbox={bbox}")
    return out


def main(args: argparse.Namespace) -> None:
    pbf = Path(args.pbf)
    bbox: BBox = _parse_bbox(args.bbox)
    sample_bbox = _shrink_bbox(bbox, float(args.sample_margin)) if args.sample_margin > 0 else bbox
    out_dir = Path(args.out_dir)

    nodes_df, edges_df = load_osm_network(pbf, bbox, network_type=args.network_type)
    g_full = build_graph(
        nodes_df,
        edges_df,
        weight=args.weight,
        default_kmh=float(args.default_kmh),
        convert_mph=bool(args.convert_mph),
    )
    graph = largest_component(g_full)

    rng = np.random.default_rng(int(args.seed))
    pool_graph_idx = sample_nodes_in_bbox(
        graph.coords_lonlat,
        n=int(args.k),
        bbox=sample_bbox,
        rng=rng,
    )
    node_ids = np.asarray([graph.node_ids[int(i)] for i in pool_graph_idx], dtype=np.int64)
    coords_lonlat = graph.coords_lonlat[pool_graph_idx].astype(np.float32, copy=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    cost_path = out_dir / "cost_matrix.npy"
    cost_mm = numpy.lib.format.open_memmap(
        cost_path, mode="w+", dtype=np.float32, shape=(int(args.k), int(args.k))
    )

    all_pairs_dijkstra_subset(
        graph.g,
        sources_idx=pool_graph_idx,
        out=cost_mm,
        symmetric=(args.symmetrize != "none"),
        log_every=int(args.log_every),
    )
    np.fill_diagonal(cost_mm, 0.0)

    # For undirected road graphs, Dijkstra distances are symmetric. `symmetric=True` above fills both triangles.
    if args.symmetrize == "avg":
        # Optional: average the triangles to reduce any floating asymmetry (rare).
        cost_mm[:] = 0.5 * (cost_mm + cost_mm.T)
    elif args.symmetrize not in ("none", "min"):
        raise ValueError(f"Unknown symmetrize: {args.symmetrize}")

    meta: Dict[str, Any] = {
        "format_version": 1,
        "created_at": _dt.datetime.now().isoformat(),
        "pbf": str(pbf),
        "bbox": list(bbox),
        "sample_bbox": list(sample_bbox),
        "sample_margin": float(args.sample_margin),
        "network_type": str(args.network_type),
        "weight": str(args.weight),
        "symmetrize": str(args.symmetrize),
        "k": int(args.k),
        "seed": int(args.seed),
        "default_kmh": float(args.default_kmh),
        "convert_mph": bool(args.convert_mph),
        "graph_stats": graph.stats,
    }
    save_pool(out_dir, meta=meta, node_ids=node_ids, coords_lonlat=coords_lonlat, cost_matrix_path=cost_path)
    print(json.dumps({"out_dir": str(out_dir), "k": int(args.k)}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pbf", type=str, required=True)
    p.add_argument("--bbox", type=str, required=True, help="lat_min,lat_max,lon_min,lon_max")
    p.add_argument("--network_type", type=str, default="driving")
    p.add_argument("--weight", type=str, default="time", choices=["time", "distance"])
    p.add_argument("--symmetrize", type=str, default="min", choices=["none", "min", "avg"])
    p.add_argument("--default_kmh", type=float, default=35.0)
    p.add_argument("--convert_mph", action="store_true")
    p.add_argument("--k", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample_margin", type=float, default=0.005, help="Shrink bbox by this margin (degrees) before sampling nodes.")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--log_every", type=int, default=100, help="Log every N Dijkstra sources (0 disables).")
    main(p.parse_args())
