from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from tensordict import TensorDict
from rl4co.envs.common.utils import Generator

from road_tsp.pool import RoadTSPPool, load_pool


def _batch_size_to_int(batch_size: Union[int, Sequence[int], torch.Size]) -> int:
    if isinstance(batch_size, int):
        return batch_size
    if isinstance(batch_size, torch.Size):
        return int(batch_size[0]) if len(batch_size) > 0 else 1
    if isinstance(batch_size, Sequence):
        return int(batch_size[0]) if len(batch_size) > 0 else 1
    raise TypeError(f"Unsupported batch_size type: {type(batch_size)}")


@dataclass
class PoolSubmatrixConfig:
    pool_dir: Path
    num_loc: int
    seed: int = 0
    mmap: bool = True
    cost_scale: float = 1.0
    bbox_jitter: bool = False
    bbox_window_frac: float = 1.0
    bbox_max_tries: int = 50


class PoolSubmatrixGenerator(Generator):
    """Samples random induced subgraphs from a precomputed pool `K×K` cost matrix.

    Intended use:
    - Offline: build a pool from an OSM road graph (or any other source).
    - Online: repeatedly sample size-`N` subsets and slice out `N×N` costs for ATSP/TSP training.
    """

    def __init__(
        self,
        pool_dir: str | Path,
        num_loc: int,
        *,
        seed: int = 0,
        mmap: bool = True,
        cost_scale: float = 1.0,
        bbox_jitter: bool = False,
        bbox_window_frac: float = 1.0,
        bbox_max_tries: int = 50,
    ):
        super().__init__()
        self.pool_dir = Path(pool_dir)
        self.num_loc = int(num_loc)
        self.seed = int(seed)
        self.mmap = bool(mmap)
        self.cost_scale = float(cost_scale)
        self.bbox_jitter = bool(bbox_jitter)
        self.bbox_window_frac = float(bbox_window_frac)
        self.bbox_max_tries = int(bbox_max_tries)
        if not (self.cost_scale > 0):
            raise ValueError(f"cost_scale must be > 0, got {self.cost_scale}")
        if not (0.0 < self.bbox_window_frac <= 1.0):
            raise ValueError(f"bbox_window_frac must be in (0,1], got {self.bbox_window_frac}")
        if self.bbox_max_tries <= 0:
            raise ValueError(f"bbox_max_tries must be > 0, got {self.bbox_max_tries}")

        # Required by RL4CO env specs; updated once pool is loaded.
        self.min_dist = 0.0
        self.max_dist = 1e9

        self._rng = np.random.default_rng(self.seed)
        self._pool: Optional[RoadTSPPool] = None
        self._bbox_lonlat: Optional[tuple[float, float, float, float]] = None  # (lon_min, lon_max, lat_min, lat_max)

    def __getstate__(self):
        state = dict(self.__dict__)
        # Avoid pickling large memmaps into env.pkl; reload lazily from pool_dir.
        state["_pool"] = None
        state["_bbox_lonlat"] = None
        return state

    def _ensure_loaded(self) -> None:
        if self._pool is not None:
            return
        pool = load_pool(self.pool_dir, mmap=self.mmap)
        k = int(pool.cost_matrix.shape[0])
        if pool.cost_matrix.shape != (k, k):
            raise ValueError(f"Pool cost_matrix must be square, got {pool.cost_matrix.shape}")
        if pool.coords_lonlat.shape[0] != k or pool.coords_lonlat.shape[1] != 2:
            raise ValueError(f"Pool coords_lonlat must have shape (K,2), got {pool.coords_lonlat.shape}")
        if pool.node_ids.shape[0] != k:
            raise ValueError(f"Pool node_ids must have shape (K,), got {pool.node_ids.shape}")
        if self.num_loc > k:
            raise ValueError(f"Requested num_loc={self.num_loc} > pool size K={k} from {self.pool_dir}")

        # Cheap bounds estimate from random samples (avoid full nanmin/nanmax on K^2).
        sample = 2048
        ii = self._rng.integers(0, k, size=sample)
        jj = self._rng.integers(0, k, size=sample)
        vals = np.asarray(pool.cost_matrix[ii, jj], dtype=np.float32)
        finite = vals[np.isfinite(vals)]
        if finite.size > 0:
            self.min_dist = float(np.min(finite))
            self.max_dist = float(np.max(finite))

        # Pool bbox from coords (lon/lat)
        coords = np.asarray(pool.coords_lonlat, dtype=np.float32)
        lon_min = float(np.min(coords[:, 0]))
        lon_max = float(np.max(coords[:, 0]))
        lat_min = float(np.min(coords[:, 1]))
        lat_max = float(np.max(coords[:, 1]))
        self._bbox_lonlat = (lon_min, lon_max, lat_min, lat_max)

        self._pool = pool

    def _sample_indices(self, b: int) -> np.ndarray:
        """Return idxs with shape (B, N) of pool indices."""
        assert self._pool is not None
        k = int(self._pool.cost_matrix.shape[0])

        if not self.bbox_jitter or self.bbox_window_frac >= 1.0:
            idxs = np.empty((b, self.num_loc), dtype=np.int64)
            for i in range(b):
                idxs[i] = self._rng.choice(k, size=self.num_loc, replace=False)
            return idxs

        assert self._bbox_lonlat is not None
        lon_min, lon_max, lat_min, lat_max = self._bbox_lonlat
        lon_span = max(lon_max - lon_min, 1e-9)
        lat_span = max(lat_max - lat_min, 1e-9)
        win_lon = lon_span * self.bbox_window_frac
        win_lat = lat_span * self.bbox_window_frac
        lon0_max = lon_max - win_lon
        lat0_max = lat_max - win_lat

        coords = np.asarray(self._pool.coords_lonlat, dtype=np.float32)
        idxs = np.empty((b, self.num_loc), dtype=np.int64)
        for bi in range(b):
            chosen = None
            for _ in range(self.bbox_max_tries):
                lon0 = float(self._rng.uniform(lon_min, lon0_max))
                lat0 = float(self._rng.uniform(lat_min, lat0_max))
                lon1 = lon0 + win_lon
                lat1 = lat0 + win_lat
                mask = (coords[:, 0] >= lon0) & (coords[:, 0] <= lon1) & (coords[:, 1] >= lat0) & (coords[:, 1] <= lat1)
                candidates = np.nonzero(mask)[0]
                if candidates.size >= self.num_loc:
                    chosen = self._rng.choice(candidates, size=self.num_loc, replace=False)
                    break
            if chosen is None:
                # Fallback: uniform over whole pool (avoid rare failures when window is too small / sparse).
                chosen = self._rng.choice(k, size=self.num_loc, replace=False)
            idxs[bi] = chosen
        return idxs

    def sample_with_meta(self, batch_size: int) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Sample a batch and also return selected pool indices and lon/lat coords."""
        self._ensure_loaded()
        assert self._pool is not None

        b = int(batch_size)
        idxs = self._sample_indices(b)

        # Vectorized advanced indexing:
        # sub[b,i,j] = cost[idxs[b,i], idxs[b,j]]  -> shape (B,N,N)
        sub = np.asarray(
            self._pool.cost_matrix[idxs[:, :, None], idxs[:, None, :]],  # type: ignore[index]
            dtype=np.float32,
        )
        if self.cost_scale != 1.0:
            sub = sub / self.cost_scale
        diag = np.arange(self.num_loc, dtype=np.int64)
        sub[:, diag, diag] = 0.0
        coords = self._pool.coords_lonlat[idxs].astype(np.float32, copy=False)
        return torch.from_numpy(sub), idxs, coords

    def _generate(self, batch_size: Union[int, Sequence[int], torch.Size]) -> TensorDict:
        self._ensure_loaded()
        assert self._pool is not None

        b = _batch_size_to_int(batch_size)
        if b <= 0:
            raise ValueError(f"batch_size must be positive, got {b}")

        idxs = self._sample_indices(b)

        mats = np.asarray(
            self._pool.cost_matrix[idxs[:, :, None], idxs[:, None, :]],  # type: ignore[index]
            dtype=np.float32,
        )
        if self.cost_scale != 1.0:
            mats = mats / self.cost_scale
        diag = np.arange(self.num_loc, dtype=np.int64)
        mats[:, diag, diag] = 0.0
        cost_matrix = torch.from_numpy(mats)
        return TensorDict({"cost_matrix": cost_matrix}, batch_size=[b])
