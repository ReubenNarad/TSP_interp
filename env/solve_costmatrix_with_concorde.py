from __future__ import annotations

import argparse
import os
import pickle
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch


SymmetrizeMode = Literal["none", "min", "avg"]


def write_tsplib_full_matrix(path: Path, cost: np.ndarray, *, display_coords: Optional[np.ndarray] = None) -> None:
    """Write a symmetric FULL_MATRIX TSPLIB file.

    Concorde is a symmetric TSP solver; if `cost` is not symmetric, you must
    symmetrize before calling this.
    """
    n = int(cost.shape[0])
    if cost.shape != (n, n):
        raise ValueError(f"Expected square cost matrix, got {cost.shape}")

    mat = np.asarray(cost, dtype=np.float64)
    mat = np.nan_to_num(mat, nan=1e12, posinf=1e12, neginf=1e12)
    mat[mat < 0] = 1e12
    np.fill_diagonal(mat, 0.0)
    mat = np.rint(mat).astype(np.int64)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"NAME: {path.stem}\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {n}\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        for row in mat:
            f.write(" ".join(str(int(x)) for x in row) + "\n")
        if display_coords is not None and display_coords.shape == (n, 2):
            f.write("DISPLAY_DATA_TYPE: TWOD_DISPLAY\n")
            f.write("DISPLAY_DATA_SECTION\n")
            for idx, (x, y) in enumerate(display_coords, start=1):
                f.write(f"{idx} {float(x):.6f} {float(y):.6f}\n")
        f.write("EOF\n")


def run_concorde(instance_filename: Path, solution_filename: Path) -> int:
    """Run Concorde and return parsed 'Number of bbnodes' if present, else -1."""
    cwd = instance_filename.parent
    bb_nodes = -1
    try:
        result = subprocess.run(
            ["concorde", "-o", solution_filename.name, instance_filename.name],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd),
        )
        match = re.search(r"Number of bbnodes:\s*(\d+)", result.stdout)
        if match:
            bb_nodes = int(match.group(1))
        else:
            match = re.search(r"Number of bbnodes:\s*(\d+)", result.stderr)
            if match:
                bb_nodes = int(match.group(1))
    except subprocess.CalledProcessError as e:
        print(f"Error executing Concorde for {instance_filename}:")
        print(e.stderr)
        match = re.search(r"Number of bbnodes:\s*(\d+)", e.stderr)
        if match:
            bb_nodes = int(match.group(1))
    return bb_nodes


def read_solution_file(filename: Path) -> list[int]:
    """Read Concorde .sol file and return a 0-based tour list."""
    tour: list[int] = []
    lines = filename.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines):
        if i == 0:
            continue
        nums = [int(x) for x in line.strip().split()]
        for num in nums:
            if num == -1:
                break
            tour.append(num)
    # Concorde uses 0-based in .sol (but TSPLIB nodes are 1-based); empirically Concorde writes 0-based.
    # If it outputs 1..n, shift down.
    if tour and min(tour) == 1 and max(tour) == len(tour):
        tour = [t - 1 for t in tour]
    return tour


def symmetrize(cost: np.ndarray, mode: SymmetrizeMode) -> np.ndarray:
    if mode == "none":
        return cost
    if mode == "min":
        return np.minimum(cost, cost.T)
    if mode == "avg":
        return 0.5 * (cost + cost.T)
    raise ValueError(f"Unknown symmetrize mode: {mode}")


def _cleanup_concorde_artifacts(dir_path: Path, base_stub: str) -> None:
    """Remove Concorde intermediate artifacts for an instance base name.

    Concorde commonly writes `<base>.<ext>` and `O<base>.<ext>` into its working directory.
    """
    exts = (".pul", ".sav", ".res", ".mas", ".pix", ".sol", ".tsp")
    for ext in exts:
        for prefix in ("", "O"):
            p = dir_path / f"{prefix}{base_stub}{ext}"
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass


def tour_cost(cost: torch.Tensor, tour: torch.Tensor) -> torch.Tensor:
    # cost: [N,N], tour: [N]
    src = tour
    tgt = torch.roll(tour, -1, dims=0)
    return cost[src, tgt].sum()


def main(args: argparse.Namespace) -> None:
    run_path = Path("runs") / args.run_name
    val_td_path = run_path / "val_td.pkl"
    if not val_td_path.exists():
        raise FileNotFoundError(f"Missing {val_td_path}")

    td = pickle.load(open(val_td_path, "rb"))
    if "cost_matrix" not in td:
        raise KeyError("val_td.pkl has no 'cost_matrix' key; this solver is for ATSP/cost-matrix runs.")

    cost_matrix: torch.Tensor = td["cost_matrix"]  # [B,N,N]
    b, n, _ = cost_matrix.shape
    num = b if args.max_instances is None else min(b, int(args.max_instances))
    print(f"Solving {num} instances of size N={n} with Concorde...")

    # simple display coords for Concorde tooling (optional)
    theta = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
    display = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)

    actions = torch.empty((num, n), dtype=torch.long)
    rewards = torch.empty((num,), dtype=torch.float32)
    bb_nodes = torch.empty((num,), dtype=torch.long)

    with tempfile.TemporaryDirectory(prefix="concorde_", dir=str(run_path)) as tmp:
        scratch = Path(tmp)
        for i in range(num):
            cm = cost_matrix[i].cpu().numpy().astype(np.float64, copy=False)
            cm = symmetrize(cm, args.symmetrize)
            cm[np.arange(n), np.arange(n)] = 0.0
            tsp_path = scratch / f"instance_{i}.tsp"
            sol_path = scratch / f"solution_{i}.sol"
            write_tsplib_full_matrix(tsp_path, cm, display_coords=display)

            bb = run_concorde(tsp_path, sol_path)
            tour = read_solution_file(sol_path)
            if len(tour) != n:
                raise ValueError(f"Concorde returned tour length {len(tour)} for instance {i} (expected {n})")

            tour_t = torch.tensor(tour, dtype=torch.long)
            c = tour_cost(cost_matrix[i], tour_t).float()
            actions[i] = tour_t
            rewards[i] = -c
            bb_nodes[i] = int(bb)

            # Concorde can emit intermediate artifacts; delete per-instance as we go.
            _cleanup_concorde_artifacts(scratch, f"instance_{i}")
            _cleanup_concorde_artifacts(scratch, f"solution_{i}")

    out = {
        "actions": [actions],
        "rewards": [rewards],
        "bb_nodes": [bb_nodes],
        "symmetrize": args.symmetrize,
    }
    out_path = run_path / args.output_name
    with open(out_path, "wb") as f:
        pickle.dump(out, f)

    avg_cost = (-rewards).mean().item()
    avg_nodes = bb_nodes[bb_nodes != -1].float().mean().item() if (bb_nodes != -1).any() else float("nan")
    print(f"Wrote {out_path} | avg optimal cost: {avg_cost:.2f} | avg bbnodes: {avg_nodes:.1f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--output_name", type=str, default="concorde_baseline.pkl")
    p.add_argument("--max_instances", type=int, default=None)
    p.add_argument("--symmetrize", type=str, choices=["none", "min", "avg"], default="min")
    main(p.parse_args())
