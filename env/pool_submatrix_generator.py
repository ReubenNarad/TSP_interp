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


class PoolSubmatrixGenerator(Generator):
    """Samples random induced subgraphs from a precomputed pool `K×K` cost matrix.

    Intended use:
    - Offline: build a pool from an OSM road graph (or any other source).
    - Online: repeatedly sample size-`N` subsets and slice out `N×N` costs for ATSP/TSP training.
    """

    def __init__(self, pool_dir: str | Path, num_loc: int, *, seed: int = 0, mmap: bool = True):
        super().__init__()
        self.pool_dir = Path(pool_dir)
        self.num_loc = int(num_loc)
        self.seed = int(seed)
        self.mmap = bool(mmap)

        # Required by RL4CO env specs; updated once pool is loaded.
        self.min_dist = 0.0
        self.max_dist = 1e9

        self._rng = np.random.default_rng(self.seed)
        self._pool: Optional[RoadTSPPool] = None

    def __getstate__(self):
        state = dict(self.__dict__)
        # Avoid pickling large memmaps into env.pkl; reload lazily from pool_dir.
        state["_pool"] = None
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

        self._pool = pool

    def sample_with_meta(self, batch_size: int) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Sample a batch and also return selected pool indices and lon/lat coords."""
        self._ensure_loaded()
        assert self._pool is not None

        k = int(self._pool.cost_matrix.shape[0])
        b = int(batch_size)
        mats = np.empty((b, self.num_loc, self.num_loc), dtype=np.float32)
        idxs = np.empty((b, self.num_loc), dtype=np.int64)
        coords = np.empty((b, self.num_loc, 2), dtype=np.float32)

        for i in range(b):
            idx = self._rng.choice(k, size=self.num_loc, replace=False)
            idxs[i] = idx
            sub = np.asarray(self._pool.cost_matrix[np.ix_(idx, idx)], dtype=np.float32)
            np.fill_diagonal(sub, 0.0)
            mats[i] = sub
            coords[i] = self._pool.coords_lonlat[idx].astype(np.float32, copy=False)

        return torch.from_numpy(mats), idxs, coords

    def _generate(self, batch_size: Union[int, Sequence[int], torch.Size]) -> TensorDict:
        self._ensure_loaded()
        assert self._pool is not None

        b = _batch_size_to_int(batch_size)
        if b <= 0:
            raise ValueError(f"batch_size must be positive, got {b}")

        k = int(self._pool.cost_matrix.shape[0])
        mats = np.empty((b, self.num_loc, self.num_loc), dtype=np.float32)
        for i in range(b):
            idx = self._rng.choice(k, size=self.num_loc, replace=False)
            sub = np.asarray(self._pool.cost_matrix[np.ix_(idx, idx)], dtype=np.float32)
            np.fill_diagonal(sub, 0.0)
            mats[i] = sub

        cost_matrix = torch.from_numpy(mats)
        return TensorDict({"cost_matrix": cost_matrix}, batch_size=[b])

