import argparse
import json
import pickle
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree

from env.osm_tools_min import build_graph, largest_component, load_osm_network


def _circle_coords(n: int) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
    return np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)


def _normalize_coords_xy(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    mn = coords.min(axis=0, keepdims=True)
    mx = coords.max(axis=0, keepdims=True)
    denom = np.where((mx - mn) > 0, (mx - mn), 1.0)
    return (coords - mn) / denom


def render_tour(
    ax,
    coords: np.ndarray,
    tour: np.ndarray,
    *,
    title: str,
    use_lonlat_axes: bool,
    pad: float,
    snap_graph=None,
    city_to_graph: np.ndarray | None = None,
    path_cache: dict[tuple[int, int], np.ndarray] | None = None,
):
    pts = coords
    closed = np.concatenate([tour, tour[:1]])

    if use_lonlat_axes and snap_graph is not None and city_to_graph is not None and path_cache is not None:
        import networkit as nk  # type: ignore

        g = snap_graph.g
        coords_lonlat = snap_graph.coords_lonlat
        lines = []
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
                    poly = np.stack([pts[int(a)], pts[int(b)]], axis=0).astype(np.float32)
                else:
                    poly = coords_lonlat[np.asarray(path_idx, dtype=int)]
                path_cache[key] = poly
            (ln,) = ax.plot(poly[:, 0], poly[:, 1], color="tab:red", linewidth=1.1, alpha=0.85, zorder=4)
            lines.append(ln)
    else:
        path = pts[closed]
        (ln,) = ax.plot(path[:, 0], path[:, 1], color="tab:red", linewidth=1.2, alpha=0.85, zorder=4)
        lines = [ln]

    start = pts[int(tour[0])]
    ax.set_title(title)

    (start_scatter,) = ax.plot(
        [start[0]],
        [start[1]],
        linestyle="none",
        marker="o",
        markersize=8,
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=1.2,
        zorder=5,
    )

    if use_lonlat_axes:
        pass
    else:
        ax.set_aspect("equal")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.axis("off")
    return lines, start_scatter


def main(args):
    run_path = Path("runs") / args.run_name
    config = json.loads((run_path / "config.json").read_text())
    num_epochs = int(args.num_epochs) if args.num_epochs is not None else int(config.get("num_epochs", 0))

    coords_path = run_path / "val_coords_lonlat.npy"
    use_lonlat_axes = False
    osm_pbf = Path(args.pbf) if args.pbf else None
    tsplib_path = Path(config["tsplib_path"]) if "tsplib_path" in config and config["tsplib_path"] else None
    pool_dir = Path(config["pool_dir"]) if "pool_dir" in config and config["pool_dir"] else None
    nodes_json_path = None
    nodes_payload = None
    if tsplib_path is not None:
        candidate = tsplib_path.with_suffix(".nodes.json")
        if candidate.exists():
            nodes_json_path = candidate
            try:
                nodes_payload = json.loads(candidate.read_text())
            except Exception:
                nodes_payload = None

    pool_meta = None
    pool_node_ids = None
    if pool_dir is not None:
        meta_path = pool_dir / "meta.json"
        node_ids_path = pool_dir / "node_ids.npy"
        if meta_path.exists():
            try:
                pool_meta = json.loads(meta_path.read_text())
            except Exception:
                pool_meta = None
        if node_ids_path.exists():
            try:
                pool_node_ids = np.load(node_ids_path)
            except Exception:
                pool_node_ids = None

    if coords_path.exists():
        coords = np.load(coords_path)  # [B,N,2] lon/lat
        coords_xy = coords[args.plot_instance].astype(np.float32, copy=False)
        use_lonlat_axes = True
    else:
        val_td = pickle.load(open(run_path / "val_td.pkl", "rb"))
        n = int(val_td["cost_matrix"].shape[1])
        coords_xy = _normalize_coords_xy(_circle_coords(n))

    results_dir = run_path / "results"
    renders_dir = run_path / "renders_matnet"
    renders_dir.mkdir(parents=True, exist_ok=True)

    images = []
    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)

    if use_lonlat_axes and osm_pbf is not None:
        network_type = "driving"
        if pool_meta is not None and pool_meta.get("network_type"):
            network_type = str(pool_meta["network_type"])
        elif nodes_payload is not None and nodes_payload.get("network_type"):
            network_type = str(nodes_payload["network_type"])

        bbox_src = None
        if pool_meta is not None and pool_meta.get("bbox") and len(pool_meta["bbox"]) == 4:
            bbox_src = pool_meta["bbox"]
        elif nodes_payload is not None and nodes_payload.get("bbox") and len(nodes_payload["bbox"]) == 4:
            bbox_src = nodes_payload["bbox"]

        if bbox_src is not None:
            lat_min_b, lat_max_b, lon_min_b, lon_max_b = [float(x) for x in bbox_src]
            lon_min, lon_max = lon_min_b - args.pad, lon_max_b + args.pad
            lat_min, lat_max = lat_min_b - args.pad, lat_max_b + args.pad
        else:
            lon_min, lon_max = float(coords_xy[:, 0].min() - args.pad), float(coords_xy[:, 0].max() + args.pad)
            lat_min, lat_max = float(coords_xy[:, 1].min() - args.pad), float(coords_xy[:, 1].max() + args.pad)

        nodes_df, edges_df = load_osm_network(osm_pbf, (lat_min, lat_max, lon_min, lon_max), network_type=network_type)
        g_full = build_graph(nodes_df, edges_df, weight="time", default_kmh=35.0, convert_mph=False)
        snap_graph = largest_component(g_full)

        # OSM background segments (vector), like GEPA_TSP plots.
        try:
            from shapely.geometry import LineString, MultiLineString  # type: ignore

            segments = []
            for geom in edges_df["geometry"]:
                if isinstance(geom, LineString):
                    segments.append(np.asarray(geom.coords))
                elif isinstance(geom, MultiLineString):
                    for part in geom.geoms:
                        segments.append(np.asarray(part.coords))
            roads = LineCollection(segments, colors="0.82", linewidths=0.4, alpha=0.6, zorder=1)
            ax.add_collection(roads)
        except Exception:
            pass

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.grid(True, color="0.85", linewidth=0.8, alpha=0.8)

        # Static city scatter (fixed instance)
        city_scatter = ax.scatter(coords_xy[:, 0], coords_xy[:, 1], s=18, c="tab:blue", alpha=0.9, zorder=3)

        city_to_graph = None
        val_indices_path = run_path / "val_indices.npy"
        base_node_ids = None
        if pool_node_ids is not None:
            base_node_ids = pool_node_ids
        elif nodes_payload is not None:
            base_node_ids = nodes_payload.get("node_ids")

        if base_node_ids is not None and val_indices_path.exists():
            idxs = np.load(val_indices_path)[args.plot_instance].astype(np.int64)
            try:
                node_ids_sel = [int(base_node_ids[i]) for i in idxs.tolist()]
                mapped = []
                for nid in node_ids_sel:
                    gi = snap_graph.id_to_idx.get(nid)
                    mapped.append(int(gi) if gi is not None else -1)
                mapped = np.asarray(mapped, dtype=np.int64)
                if (mapped >= 0).all():
                    city_to_graph = mapped
            except Exception:
                city_to_graph = None

        if city_to_graph is None:
            tree = cKDTree(snap_graph.coords_lonlat)
            _dist, idxs = tree.query(coords_xy)
            city_to_graph = idxs.astype(np.int64)

        path_cache: dict[tuple[int, int], np.ndarray] = {}
    else:
        snap_graph = None
        city_to_graph = None
        path_cache = None

    prev_lines = []
    prev_start = None

    for epoch in range(0, num_epochs, args.step):
        res_path = results_dir / f"results_epoch_{epoch}.pkl"
        if not res_path.exists():
            continue
        res = pickle.load(open(res_path, "rb"))
        tour = np.asarray(res["actions"][args.plot_instance].cpu().numpy(), dtype=np.int64)
        reward = float(res["rewards"][args.plot_instance].item())
        cost = -reward

        title = f"Epoch {epoch} | tour cost: {cost:,.0f}"

        for ln in prev_lines:
            try:
                ln.remove()
            except Exception:
                pass
        prev_lines = []
        if prev_start is not None:
            try:
                prev_start.remove()
            except Exception:
                pass

        prev_lines, prev_start = render_tour(
            ax,
            coords_xy,
            tour,
            title=title,
            use_lonlat_axes=use_lonlat_axes,
            pad=args.pad,
            snap_graph=snap_graph,
            city_to_graph=city_to_graph,
            path_cache=path_cache,
        )
        out_png = renders_dir / f"epoch_{epoch:04d}_inst_{args.plot_instance}.png"
        fig.savefig(out_png, dpi=140, bbox_inches="tight")
        images.append(imageio.imread(out_png))

    plt.close(fig)

    if not images:
        raise RuntimeError(f"No frames rendered; check results in {results_dir}")

    gif_path = run_path / f"animation_matnet_inst_{args.plot_instance}.gif"
    imageio.mimsave(gif_path, images, fps=args.fps)
    print(f"Wrote {gif_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument("--plot_instance", type=int, default=0)
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--pbf", type=str, default=None, help="Optional .osm.pbf file to draw roads behind lon/lat coords.")
    p.add_argument("--pad", type=float, default=0.005, help="Padding for lon/lat plot bbox.")
    main(p.parse_args())
