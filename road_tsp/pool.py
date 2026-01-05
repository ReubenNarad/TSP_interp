from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]  # lat_min, lat_max, lon_min, lon_max
Weight = Literal["time", "distance"]


@dataclass(frozen=True)
class RoadTSPPool:
    pool_dir: Path
    node_ids: np.ndarray  # shape (K,), int64 OSM node ids
    coords_lonlat: np.ndarray  # shape (K,2) float32 -> [lon, lat]
    cost_matrix: np.ndarray  # shape (K,K) float32 (may be memmap)
    meta: Dict[str, Any]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def load_pool(pool_dir: str | Path, *, mmap: bool = True) -> RoadTSPPool:
    pool_dir = Path(pool_dir)
    meta = _read_json(pool_dir / "meta.json")
    node_ids = np.load(pool_dir / "node_ids.npy")
    coords_lonlat = np.load(pool_dir / "coords_lonlat.npy")
    cost_path = pool_dir / "cost_matrix.npy"
    if mmap:
        cost = np.load(cost_path, mmap_mode="r")
    else:
        cost = np.load(cost_path)
    return RoadTSPPool(pool_dir=pool_dir, node_ids=node_ids, coords_lonlat=coords_lonlat, cost_matrix=cost, meta=meta)


def save_pool(
    pool_dir: str | Path,
    *,
    meta: Dict[str, Any],
    node_ids: np.ndarray,
    coords_lonlat: np.ndarray,
    cost_matrix_path: Optional[Path] = None,
) -> None:
    """Persist pool metadata and arrays.

    `cost_matrix_path` allows callers to create/fill the cost matrix separately (e.g. via memmap).
    """
    pool_dir = Path(pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)
    _write_json(pool_dir / "meta.json", meta)
    np.save(pool_dir / "node_ids.npy", np.asarray(node_ids, dtype=np.int64))
    np.save(pool_dir / "coords_lonlat.npy", np.asarray(coords_lonlat, dtype=np.float32))
    if cost_matrix_path is not None:
        # The builder writes cost_matrix.npy directly; keep a sanity check that the name matches.
        expected = pool_dir / "cost_matrix.npy"
        if cost_matrix_path.resolve() != expected.resolve():
            raise ValueError(f"Expected cost_matrix at {expected}, got {cost_matrix_path}")

