from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union, Literal

import numpy as np
import torch
from tensordict import TensorDict
from rl4co.envs.common.utils import Generator

from .tsplib import load_tsplib


SymmetrizeMode = Literal["none", "min", "avg"]


def _batch_size_to_int(batch_size: Union[int, Sequence[int], torch.Size]) -> int:
    if isinstance(batch_size, int):
        return batch_size
    if isinstance(batch_size, torch.Size):
        return int(batch_size[0]) if len(batch_size) > 0 else 1
    if isinstance(batch_size, Sequence):
        return int(batch_size[0]) if len(batch_size) > 0 else 1
    raise TypeError(f"Unsupported batch_size type: {type(batch_size)}")


@dataclass
class TSPLIBSubmatrixConfig:
    tsp_path: Path
    num_loc: int
    symmetrize: SymmetrizeMode = "none"
    seed: int = 0


class TSPLIBSubmatrixGenerator(Generator):
    """Samples random induced subgraphs from a TSPLIB FULL_MATRIX instance.

    This is a pragmatic way to train on realistic road-network costs without
    recomputing all-pairs shortest paths per batch.
    """

    def __init__(
        self,
        tsp_path: str | Path,
        num_loc: int,
        *,
        symmetrize: SymmetrizeMode = "none",
        seed: int = 0,
    ):
        super().__init__()
        self.tsp_path = Path(tsp_path)
        self.num_loc = int(num_loc)
        # These are required by RL4CO's ATSPEnv spec builder.
        # They will be refined once the base matrix is loaded.
        self.min_dist = 0.0
        # Set a generous upper bound to avoid spec violations before lazy load.
        self.max_dist = 1e9
        self.symmetrize = symmetrize
        self.seed = int(seed)

        self._rng = np.random.default_rng(self.seed)
        self._base_cost: Optional[np.ndarray] = None
        self._base_n: Optional[int] = None
        self._base_coords_lonlat: Optional[np.ndarray] = None

    def __getstate__(self):
        state = dict(self.__dict__)
        # Avoid pickling the full matrix into env.pkl; reload lazily from tsp_path.
        state["_base_cost"] = None
        state["_base_coords_lonlat"] = None
        return state

    def _ensure_loaded(self) -> None:
        if self._base_cost is not None:
            return
        prob = load_tsplib(self.tsp_path)
        if prob.cost_matrix is None:
            raise ValueError(f"TSPLIB file has no FULL_MATRIX cost section: {self.tsp_path}")
        self._base_cost = prob.cost_matrix
        self._base_n = int(prob.dimension)
        self._base_coords_lonlat = prob.display_coords_lonlat
        # Update distance bounds for a more accurate observation spec.
        self.min_dist = float(np.nanmin(self._base_cost))
        self.max_dist = float(np.nanmax(self._base_cost))
        if self.num_loc > self._base_n:
            raise ValueError(f"Requested num_loc={self.num_loc} > base dimension {self._base_n} from {self.tsp_path}")

    def sample_with_meta(self, batch_size: int) -> tuple[torch.Tensor, np.ndarray, Optional[np.ndarray]]:
        """Sample a batch and also return the selected base indices and coords (if present)."""
        self._ensure_loaded()
        assert self._base_cost is not None
        assert self._base_n is not None

        b = int(batch_size)
        mats = np.empty((b, self.num_loc, self.num_loc), dtype=np.float32)
        idxs = np.empty((b, self.num_loc), dtype=np.int64)
        coords: Optional[np.ndarray] = None
        if self._base_coords_lonlat is not None and self._base_coords_lonlat.shape[0] == self._base_n:
            coords = np.empty((b, self.num_loc, 2), dtype=np.float32)

        for i in range(b):
            idx = self._rng.choice(self._base_n, size=self.num_loc, replace=False)
            idxs[i] = idx
            sub = self._base_cost[np.ix_(idx, idx)].astype(np.float32, copy=False)
            if self.symmetrize == "min":
                sub = np.minimum(sub, sub.T)
            elif self.symmetrize == "avg":
                sub = 0.5 * (sub + sub.T)
            elif self.symmetrize != "none":
                raise ValueError(f"Unknown symmetrize mode: {self.symmetrize}")
            np.fill_diagonal(sub, 0.0)
            mats[i] = sub
            if coords is not None:
                coords[i] = self._base_coords_lonlat[idx].astype(np.float32, copy=False)

        return torch.from_numpy(mats), idxs, coords

    def _generate(self, batch_size: Union[int, Sequence[int], torch.Size]) -> TensorDict:
        self._ensure_loaded()
        assert self._base_cost is not None
        assert self._base_n is not None

        b = _batch_size_to_int(batch_size)
        if b <= 0:
            raise ValueError(f"batch_size must be positive, got {b}")

        mats = np.empty((b, self.num_loc, self.num_loc), dtype=np.float32)
        for i in range(b):
            idx = self._rng.choice(self._base_n, size=self.num_loc, replace=False)
            sub = self._base_cost[np.ix_(idx, idx)].astype(np.float32, copy=False)
            if self.symmetrize == "min":
                sub = np.minimum(sub, sub.T)
            elif self.symmetrize == "avg":
                sub = 0.5 * (sub + sub.T)
            elif self.symmetrize != "none":
                raise ValueError(f"Unknown symmetrize mode: {self.symmetrize}")
            np.fill_diagonal(sub, 0.0)
            mats[i] = sub

        cost_matrix = torch.from_numpy(mats)
        return TensorDict({"cost_matrix": cost_matrix}, batch_size=[b])
