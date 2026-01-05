from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np


EdgeWeightType = Literal["EXPLICIT", "EUC_2D"]
EdgeWeightFormat = Literal["FULL_MATRIX"]


@dataclass(frozen=True)
class TSPLIBProblem:
    name: str
    dimension: int
    edge_weight_type: EdgeWeightType
    edge_weight_format: Optional[EdgeWeightFormat]
    cost_matrix: Optional[np.ndarray]  # shape (n, n), float32
    display_coords_lonlat: Optional[np.ndarray]  # shape (n, 2), float32


def _parse_kv(line: str) -> tuple[str, str]:
    if ":" not in line:
        raise ValueError(f"Invalid TSPLIB header line (missing ':'): {line!r}")
    k, v = line.split(":", 1)
    return k.strip().upper(), v.strip()


def load_tsplib(path: str | Path) -> TSPLIBProblem:
    """Load a small subset of TSPLIB formats used in this project.

    Supported:
    - EDGE_WEIGHT_TYPE=EXPLICIT with EDGE_WEIGHT_FORMAT=FULL_MATRIX
      (parses EDGE_WEIGHT_SECTION into a dense matrix)
    - DISPLAY_DATA_SECTION as lon/lat pairs (optional)

    This intentionally does not attempt full TSPLIB coverage.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()

    name = path.stem
    dimension: Optional[int] = None
    edge_weight_type: Optional[str] = None
    edge_weight_format: Optional[str] = None

    edge_numbers: list[float] = []
    display_coords: list[tuple[float, float]] = []

    section: Optional[str] = None

    for raw in text:
        line = raw.strip()
        if not line:
            continue

        upper = line.upper()
        if upper == "EDGE_WEIGHT_SECTION":
            section = "EDGE_WEIGHT_SECTION"
            continue
        if upper == "DISPLAY_DATA_SECTION":
            section = "DISPLAY_DATA_SECTION"
            continue
        if upper == "NODE_COORD_SECTION":
            section = "NODE_COORD_SECTION"
            continue
        if upper == "EOF":
            break

        if section is None:
            k, v = _parse_kv(line)
            if k == "NAME":
                name = v
            elif k == "DIMENSION":
                dimension = int(v)
            elif k == "EDGE_WEIGHT_TYPE":
                edge_weight_type = v.upper()
            elif k == "EDGE_WEIGHT_FORMAT":
                edge_weight_format = v.upper()
            continue

        if section == "EDGE_WEIGHT_SECTION":
            edge_numbers.extend(float(x) for x in line.split())
            continue

        if section in {"DISPLAY_DATA_SECTION", "NODE_COORD_SECTION"}:
            parts = line.split()
            if len(parts) >= 3:
                lon = float(parts[1])
                lat = float(parts[2])
                display_coords.append((lon, lat))
            continue

    if dimension is None:
        raise ValueError(f"Missing DIMENSION in TSPLIB file: {path}")
    if edge_weight_type is None:
        raise ValueError(f"Missing EDGE_WEIGHT_TYPE in TSPLIB file: {path}")

    cost_matrix: Optional[np.ndarray] = None
    if edge_weight_type == "EXPLICIT":
        if edge_weight_format != "FULL_MATRIX":
            raise ValueError(
                f"Unsupported EDGE_WEIGHT_FORMAT={edge_weight_format!r} in {path} (expected FULL_MATRIX)"
            )
        expected = dimension * dimension
        if len(edge_numbers) < expected:
            raise ValueError(
                f"EDGE_WEIGHT_SECTION has {len(edge_numbers)} numbers but expected {expected} for {dimension}x{dimension}"
            )
        arr = np.asarray(edge_numbers[:expected], dtype=np.float32)
        cost_matrix = arr.reshape((dimension, dimension))

    coords_arr: Optional[np.ndarray] = None
    if display_coords:
        if len(display_coords) != dimension:
            # Some TSPLIBs omit display coords; if present, expect full length.
            coords_arr = None
        else:
            coords_arr = np.asarray(display_coords, dtype=np.float32)

    return TSPLIBProblem(
        name=name,
        dimension=dimension,
        edge_weight_type=edge_weight_type,  # type: ignore[arg-type]
        edge_weight_format=edge_weight_format,  # type: ignore[arg-type]
        cost_matrix=cost_matrix,
        display_coords_lonlat=coords_arr,
    )

