import json
import math
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from matplotlib.colors import Normalize

STATIC_DIR = Path(__file__).parent / "viewer_static"


def discover_manifests(search_root: Path) -> List[Path]:
    manifests: List[Path] = []
    if (search_root / "clt" / "clt_runs").exists():
        policy_dirs = [search_root]
    else:
        if not search_root.exists():
            return []
        policy_dirs = [
            p for p in search_root.iterdir() if (p / "clt" / "clt_runs").exists()
        ]

    for policy_dir in policy_dirs:
        clt_root = policy_dir / "clt" / "clt_runs"
        if not clt_root.exists():
            continue
        for pair_dir in sorted([p for p in clt_root.iterdir() if p.is_dir()]):
            for clt_dir in sorted([p for p in pair_dir.iterdir() if p.is_dir()]):
                manifest = clt_dir / "viz" / "viewer_data" / "manifest.json"
                if manifest.exists():
                    manifests.append(manifest)
    manifests.sort()
    return manifests


def load_manifest(path: Path) -> Dict:
    with open(path, "r") as fp:
        data = json.load(fp)
    data["manifest_path"] = str(path)
    return data


def load_dataset(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        data = {key: arrays[key] for key in arrays.files}
    actions = data.get("actions")
    if actions is not None:
        if actions.size == 0 or np.all(actions == -1):
            data["actions"] = None
    return data


def draw_tour(
    ax,
    coords: np.ndarray,
    actions: Optional[np.ndarray],
    color="black",
    alpha=0.35,
    linewidth=1.0,
) -> None:
    if actions is None:
        return
    valid = actions[actions >= 0]
    if valid.size == 0:
        return
    tour = np.concatenate([valid, valid[:1]])
    for start, end in zip(tour[:-1], tour[1:]):
        dx = coords[end, 0] - coords[start, 0]
        dy = coords[end, 1] - coords[start, 1]
        ax.arrow(
            coords[start, 0],
            coords[start, 1],
            dx,
            dy,
            head_width=0.01,
            head_length=0.02,
            fc=color,
            ec=color,
            alpha=alpha,
            linewidth=linewidth,
            length_includes_head=True,
            zorder=1,
        )


def draw_edge_intensity_tour(
    ax,
    coords: np.ndarray,
    actions: Optional[np.ndarray],
    node_values: np.ndarray,
    bold_threshold: float,
    cmap_name: str = "cividis",
) -> None:
    if actions is None:
        return
    valid = actions[actions >= 0]
    if valid.size == 0:
        return
    tour = np.concatenate([valid, valid[:1]])
    vmax = max(float(node_values.max()), 1e-6)
    cmap = plt.get_cmap(cmap_name)

    for start, end in zip(tour[:-1], tour[1:]):
        value = max(float(node_values[start]), float(node_values[end]))
        norm_val = min(max(value / vmax, 0.0), 1.0)
        bold = value >= bold_threshold if bold_threshold > 0 else False
        base_color = cmap(norm_val)
        # Less intense edges (low norm_val) stay more opaque
        alpha = 0.35 + 0.6 * (1.0 - norm_val)
        rgba = (base_color[0], base_color[1], base_color[2], alpha)
        outline = "#111111" if bold else rgba
        dx = coords[end, 0] - coords[start, 0]
        dy = coords[end, 1] - coords[start, 1]
        ax.arrow(
            coords[start, 0],
            coords[start, 1],
            dx,
            dy,
            head_width=0.012 + 0.006 * norm_val,
            head_length=0.02 + 0.01 * norm_val,
            fc=rgba,
            ec=outline,
            linewidth=0.3 + 1.2 * norm_val,
            length_includes_head=True,
            zorder=1.5 + norm_val,
        )


def render_overlay(
    locs: np.ndarray,
    latents: np.ndarray,
    actions: Optional[np.ndarray],
    feature_idx: int,
    instance_indices: List[int],
    tour_mode: str,
    tour_threshold: float,
    edge_bold_threshold: float,
    color_mode: str,
) -> bytes:
    if not instance_indices:
        raise ValueError("At least one instance index is required.")

    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
    disable_color = color_mode == "none"
    if disable_color:
        vmax = 1e-6
    else:
        values = [latents[idx, :, feature_idx] for idx in instance_indices]
        vmax = max(float(v.max()) for v in values)
        vmax = max(vmax, 1e-6)
    norm = Normalize(vmin=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    scatter = None

    for order, idx in enumerate(instance_indices):
        coords = locs[idx]
        if disable_color:
            acts = np.zeros(locs[idx].shape[0], dtype=float)
        else:
            acts = latents[idx, :, feature_idx]
        marker = markers[order % len(markers)]
        if tour_mode == "edge":
            point_alpha = 0.35
        elif tour_mode == "threshold":
            point_alpha = 0.75
        else:
            point_alpha = 0.9
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=acts,
            cmap="viridis",
            s=55,
            edgecolors="black",
            linewidths=0.35,
            marker=marker,
            norm=norm,
            alpha=point_alpha,
            zorder=2,
            label=f"Instance {idx:02d}",
        )
        if actions is None or idx >= actions.shape[0]:
            continue
        if tour_mode == "none":
            continue
        if tour_mode == "threshold":
            mean_val = float(acts.mean())
            high = mean_val >= tour_threshold
            color = "#1f77b4" if high else "#d62728"
            alpha = 0.6 if high else 0.2
            draw_tour(ax, coords, actions[idx], color=color, alpha=alpha, linewidth=1.35)
        elif tour_mode == "edge":
            draw_edge_intensity_tour(ax, coords, actions[idx], acts, edge_bold_threshold)
        else:
            draw_tour(ax, coords, actions[idx], color="black", alpha=0.25)

    if scatter is not None and tour_mode != "edge" and not disable_color:
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"Feature {feature_idx} activation")

    if disable_color:
        ax.set_title("Tours only overlay")
    else:
        ax.set_title(f"Feature {feature_idx} overlay")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    if scatter is not None:
        legend_cols = 3
        legend_rows = max(1, math.ceil(len(instance_indices) / legend_cols))
        legend_y = 1.08 + 0.08 * legend_rows
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=legend_cols,
            fontsize=8,
            frameon=False,
        )
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


def render_single(
    locs: np.ndarray,
    latents: np.ndarray,
    actions: Optional[np.ndarray],
    feature_idx: int,
    instance_idx: int,
    ref_instance_indices: List[int],
    tour_mode: str,
    tour_threshold: float,
    edge_bold_threshold: float,
    color_mode: str,
) -> bytes:
    coords = locs[instance_idx]
    disable_color = color_mode == "none"
    if disable_color:
        acts = np.zeros(coords.shape[0], dtype=float)
        vmax = 1e-6
    else:
        acts = latents[instance_idx, :, feature_idx]
        ref_vmax = max(float(latents[idx, :, feature_idx].max()) for idx in ref_instance_indices)
        vmax = max(ref_vmax, 1e-6)
    norm = Normalize(vmin=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=acts,
        cmap="viridis",
        s=120,
        edgecolors="black",
        linewidths=0.45,
        norm=norm,
    )
    if actions is not None and instance_idx < actions.shape[0] and tour_mode != "none":
        if tour_mode == "threshold":
            mean_val = float(acts.mean())
            high = mean_val >= tour_threshold
            color = "#1f77b4" if high else "#d62728"
            alpha = 0.65 if high else 0.2
            draw_tour(ax, coords, actions[instance_idx], color=color, alpha=alpha, linewidth=1.45)
        elif tour_mode == "edge":
            draw_edge_intensity_tour(ax, coords, actions[instance_idx], acts, edge_bold_threshold)
        else:
            draw_tour(ax, coords, actions[instance_idx], color="black", alpha=0.4)

    if not disable_color:
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04).set_label(
            f"Feature {feature_idx} activation"
        )
        ax.set_title(f"Instance {instance_idx:02d} — Feature {feature_idx}")
    else:
        ax.set_title(f"Instance {instance_idx:02d} — Tours only")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


class CLTRegistry:
    def __init__(self, search_root: Path, manifest_override: Optional[Path] = None):
        self.search_root = search_root
        self.manifest_override = manifest_override
        self.runs: Dict[str, Dict] = {}
        self._build_registry()

    def _build_registry(self) -> None:
        manifest_paths: List[Path]
        if self.manifest_override:
            manifest_paths = [self.manifest_override]
        else:
            manifest_paths = discover_manifests(self.search_root)

        for idx, manifest_path in enumerate(manifest_paths):
            data = load_manifest(manifest_path)
            manifest_dir = manifest_path.parent
            data_file = manifest_dir / data["data_file"]
            if not data_file.exists():
                continue
            dataset = load_dataset(data_file)
            run_id = f"run_{idx:03d}"
            label = f"{data['policy_run']} | {data['pair_name']} | {data['clt_run']}"
            self.runs[run_id] = {
                "id": run_id,
                "label": label,
                "manifest": data,
                "dataset": dataset,
            }

    def list_runs(self) -> List[Dict]:
        return [
            {
                "id": info["id"],
                "label": info["label"],
                "policy_run": info["manifest"]["policy_run"],
                "pair_name": info["manifest"]["pair_name"],
                "clt_run": info["manifest"]["clt_run"],
                "num_instances": info["manifest"]["num_instances"],
                "latent_dim": info["manifest"]["latent_dim"],
            }
            for info in self.runs.values()
        ]

    def get(self, run_id: str) -> Dict:
        if run_id not in self.runs:
            raise KeyError(run_id)
        return self.runs[run_id]


def build_registry() -> CLTRegistry:
    search_root = Path(os.environ.get("CLT_VIEWER_SEARCH_ROOT", "runs")).resolve()
    manifest_override = os.environ.get("CLT_VIEWER_MANIFEST")
    manifest_path = Path(manifest_override).resolve() if manifest_override else None
    return CLTRegistry(search_root, manifest_path)


REGISTRY = build_registry()
app = FastAPI(title="CLT Feature Viewer")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def get_run_or_404(run_id: str) -> Dict:
    try:
        return REGISTRY.get(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Viewer assets missing.")
    return index_path.read_text()


@app.get("/api/runs")
def list_runs():
    runs = REGISTRY.list_runs()
    if not runs:
        return []
    return runs


@app.get("/api/runs/{run_id}")
def run_details(run_id: str):
    info = get_run_or_404(run_id)
    manifest = info["manifest"]
    summary = {
        "id": info["id"],
        "label": info["label"],
        "policy_run": manifest["policy_run"],
        "pair_name": manifest["pair_name"],
        "clt_run": manifest["clt_run"],
        "num_instances": manifest["num_instances"],
        "latent_dim": manifest["latent_dim"],
        "top_features": manifest["feature_stats"].get("top_features", []),
        "has_actions": manifest.get("has_actions", False),
        "options": manifest.get("options", {}),
    }
    return summary


@app.get("/api/runs/{run_id}/feature_stats/{feature_idx}")
def feature_stats(run_id: str, feature_idx: int):
    info = get_run_or_404(run_id)
    manifest = info["manifest"]
    latent_dim = manifest["latent_dim"]
    if feature_idx < 0 or feature_idx >= latent_dim:
        raise HTTPException(status_code=400, detail="Feature index out of range.")
    stats = manifest["feature_stats"]
    return {
        "feature_idx": feature_idx,
        "mean_activation": stats["mean_activation"][feature_idx],
        "mean_abs_activation": stats["mean_abs_activation"][feature_idx],
        "nonzero_rate": stats["nonzero_rate"][feature_idx],
    }


@app.get("/api/runs/{run_id}/instance_means/{feature_idx}")
def instance_means(run_id: str, feature_idx: int):
    info = get_run_or_404(run_id)
    manifest = info["manifest"]
    latent_dim = manifest["latent_dim"]
    if feature_idx < 0 or feature_idx >= latent_dim:
        raise HTTPException(status_code=400, detail="Feature index out of range.")
    latents = info["dataset"]["latents"]
    means = latents[:, :, feature_idx].mean(axis=1)
    rows = [
        {"instance": idx, "mean_activation": float(means[idx])}
        for idx in range(latents.shape[0])
    ]
    return rows


@app.get("/api/runs/{run_id}/plot/overlay")
def plot_overlay(
    run_id: str,
    feature_idx: int = Query(..., ge=0),
    instances: Optional[str] = Query(None),
    show_tour: bool = Query(True),
    tour_mode: str = Query("standard"),
    tour_threshold: float = Query(0.0),
    edge_bold_threshold: float = Query(0.0),
    color_mode: str = Query("feature"),
):
    info = get_run_or_404(run_id)
    manifest = info["manifest"]
    latent_dim = manifest["latent_dim"]
    if feature_idx < 0 or feature_idx >= latent_dim:
        raise HTTPException(status_code=400, detail="Feature index out of range.")

    num_instances = manifest["num_instances"]
    if instances:
        try:
            idxs = [int(x) for x in instances.split(",") if x != ""]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid instance list.")
    else:
        default = min(6, num_instances)
        idxs = list(range(default))

    if not idxs:
        raise HTTPException(status_code=400, detail="No instances selected.")
    for idx in idxs:
        if idx < 0 or idx >= num_instances:
            raise HTTPException(status_code=400, detail=f"Instance {idx} out of range.")

    valid_modes = {"none", "standard", "threshold", "edge"}
    if tour_mode not in valid_modes:
        raise HTTPException(status_code=400, detail="Invalid tour_mode.")
    if not show_tour:
        tour_mode = "none"
    valid_color_modes = {"feature", "none"}
    if color_mode not in valid_color_modes:
        raise HTTPException(status_code=400, detail="Invalid color_mode.")

    locs = info["dataset"]["locs"]
    latents = info["dataset"]["latents"]
    actions = info["dataset"].get("actions")
    image_bytes = render_overlay(
        locs,
        latents,
        actions,
        feature_idx,
        idxs,
        tour_mode,
        tour_threshold,
        edge_bold_threshold,
        color_mode,
    )
    return Response(content=image_bytes, media_type="image/png")


@app.get("/api/runs/{run_id}/plot/single")
def plot_single_route(
    run_id: str,
    feature_idx: int = Query(..., ge=0),
    instance_idx: int = Query(0, ge=0),
    ref_instances: Optional[str] = Query(None),
    show_tour: bool = Query(True),
    tour_mode: str = Query("standard"),
    tour_threshold: float = Query(0.0),
    edge_bold_threshold: float = Query(0.0),
    color_mode: str = Query("feature"),
):
    info = get_run_or_404(run_id)
    manifest = info["manifest"]
    latent_dim = manifest["latent_dim"]
    if feature_idx < 0 or feature_idx >= latent_dim:
        raise HTTPException(status_code=400, detail="Feature index out of range.")
    if instance_idx < 0 or instance_idx >= manifest["num_instances"]:
        raise HTTPException(status_code=400, detail="Instance index out of range.")

    num_instances = manifest["num_instances"]
    ref_idxs: List[int] = []
    if ref_instances is not None:
        try:
            ref_idxs = [int(x) for x in ref_instances.split(",") if x != ""]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid reference instance list.")
    if not ref_idxs:
        ref_idxs = list(range(min(6, num_instances)))
    if instance_idx not in ref_idxs:
        ref_idxs.append(instance_idx)
    ref_idxs = sorted(set(ref_idxs))
    for idx in ref_idxs:
        if idx < 0 or idx >= num_instances:
            raise HTTPException(status_code=400, detail=f"Instance {idx} out of range.")

    valid_modes = {"none", "standard", "threshold", "edge"}
    if tour_mode not in valid_modes:
        raise HTTPException(status_code=400, detail="Invalid tour_mode.")
    if not show_tour:
        tour_mode = "none"
    valid_color_modes = {"feature", "none"}
    if color_mode not in valid_color_modes:
        raise HTTPException(status_code=400, detail="Invalid color_mode.")

    locs = info["dataset"]["locs"]
    latents = info["dataset"]["latents"]
    actions = info["dataset"].get("actions")
    image_bytes = render_single(
        locs,
        latents,
        actions,
        feature_idx,
        instance_idx,
        ref_idxs,
        tour_mode,
        tour_threshold,
        edge_bold_threshold,
        color_mode,
    )
    return Response(content=image_bytes, media_type="image/png")


def run_dev_server():
    import uvicorn
    host = os.environ.get("CLT_VIEWER_HOST", "0.0.0.0")
    port = int(os.environ.get("CLT_VIEWER_PORT", "8501"))
    uvicorn.run("clt.viewer_server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_dev_server()
