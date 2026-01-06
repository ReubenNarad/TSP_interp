#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Composite

import sys

# Ensure repo root is importable when invoked as a script.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from env.solve_costmatrix_with_concorde import read_solution_file


STATUS_OK = 0
STATUS_TIMEOUT = 1
STATUS_CALLED_PROCESS_ERROR = 2
STATUS_NO_SOLUTION_FILE = 3
STATUS_BAD_TOUR = 4
STATUS_EXCEPTION = 5

CONCORDE_INTERMEDIATE_EXTS = [".pul", ".sav", ".res", ".mas", ".pix", ".sol", ".tsp"]


@dataclass(frozen=True)
class SolveResult:
    valid: bool
    status: int
    length: float
    time_wall: float
    time_reported: float
    bb_nodes: int
    lp_rows: int
    lp_cols: int
    lp_nonzeros: int
    tour: Optional[list[int]] = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def patch_env_specs(env) -> None:
    def _patch(spec):
        if isinstance(spec, Composite):
            if not hasattr(spec, "data_cls"):
                spec.data_cls = None
            if not hasattr(spec, "step_mdp_static"):
                spec.step_mdp_static = False
            for child in spec.values():
                if child is not None:
                    _patch(child)

    for spec_name in ["input_spec", "output_spec", "observation_spec", "reward_spec"]:
        spec = getattr(env, spec_name, None)
        if spec is not None:
            _patch(spec)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def shard_bounds(num_instances_total: int, num_shards: int, shard_idx: int) -> Tuple[int, int]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_idx < 0 or shard_idx >= num_shards:
        raise ValueError(f"shard_idx must be in [0, {num_shards}), got {shard_idx}")

    base = num_instances_total // num_shards
    rem = num_instances_total % num_shards
    start = shard_idx * base + min(shard_idx, rem)
    size = base + (1 if shard_idx < rem else 0)
    end = start + size
    return start, end


def _parse_int(pattern: str, text: str) -> int:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else -1


def _parse_float(pattern: str, text: str) -> float:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else -1.0


def parse_concorde_stats(stdout: str, stderr: str) -> Tuple[int, float, int, int, int]:
    combined = "\n".join([stdout or "", stderr or ""])
    bb_nodes = _parse_int(r"Number of bbnodes:\s*(\d+)", combined)
    time_reported = _parse_float(r"Total Running Time:\s*([0-9.]+)", combined)

    lp_rows = lp_cols = lp_nonzeros = -1
    m = re.search(r"Final LP has (\d+) rows, (\d+) columns, (\d+) nonzeros", combined)
    if m:
        lp_rows, lp_cols, lp_nonzeros = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return bb_nodes, time_reported, lp_rows, lp_cols, lp_nonzeros


def cleanup_concorde_files(scratch_dir: Path, base_stub: str, keep_solution: bool = False) -> None:
    for ext in CONCORDE_INTERMEDIATE_EXTS:
        for prefix in ("", "O"):
            p = scratch_dir / f"{prefix}{base_stub}{ext}"
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
    if not keep_solution:
        sol = scratch_dir / f"{base_stub}.sol"
        if sol.exists():
            try:
                sol.unlink()
            except Exception:
                pass


def _normalize_tour_indices(route: list[int], n: int) -> Optional[list[int]]:
    if len(route) != n:
        return None
    mn, mx = min(route), max(route)
    if mn == 0 and mx == n - 1:
        route0 = route
    elif mn == 1 and mx == n:
        route0 = [x - 1 for x in route]
    else:
        return None
    if any((x < 0 or x >= n) for x in route0):
        return None
    if len(set(route0)) != n:
        return None
    return route0


def _to_int_matrix(cost: np.ndarray, scale_factor: float) -> np.ndarray:
    mat = np.asarray(cost, dtype=np.float64)
    mat = np.nan_to_num(mat, nan=1e12, posinf=1e12, neginf=1e12)
    mat[mat < 0] = 1e12
    np.fill_diagonal(mat, 0.0)
    # Use ceil to preserve triangle inequality when the underlying metric holds
    # (rounding can break it and violate monotonicity under node removal).
    mat_int = np.ceil(mat * float(scale_factor) - 1e-9).astype(np.int64)
    return mat_int


def _tour_cost_int(mat_int: np.ndarray, tour0: list[int]) -> int:
    idx = np.asarray(tour0, dtype=np.int64)
    idx_next = np.roll(idx, -1)
    return int(mat_int[idx, idx_next].sum())


def solve_with_concorde_full_matrix(
    cost: torch.Tensor,
    *,
    scratch_dir: Path,
    base_stub: str,
    scale_factor: float,
    timeout_sec: Optional[float],
) -> SolveResult:
    cost_np = cost.detach().cpu().numpy().astype(np.float64, copy=False)
    n = int(cost_np.shape[0])

    tsp_path = scratch_dir / f"{base_stub}.tsp"
    sol_path = scratch_dir / f"{base_stub}.sol"

    # Write TSPLIB FULL_MATRIX using integer weights derived from ceil(cost * scale_factor).
    mat_int = _to_int_matrix(cost_np, scale_factor=scale_factor)
    tsp_path.parent.mkdir(parents=True, exist_ok=True)
    with tsp_path.open("w", encoding="utf-8") as f:
        f.write(f"NAME: {tsp_path.stem}\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {n}\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        for row in mat_int:
            f.write(" ".join(str(int(x)) for x in row) + "\n")
        f.write("EOF\n")

    stdout = ""
    stderr = ""
    status = STATUS_OK
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            ["concorde", "-o", sol_path.name, tsp_path.name],
            cwd=str(scratch_dir),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec if timeout_sec and timeout_sec > 0 else None,
        )
        stdout, stderr = proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        status = STATUS_TIMEOUT
        stdout = (e.stdout.decode() if isinstance(e.stdout, (bytes, bytearray)) else (e.stdout or "")) or ""
        stderr = (e.stderr.decode() if isinstance(e.stderr, (bytes, bytearray)) else (e.stderr or "")) or ""
    except subprocess.CalledProcessError as e:
        status = STATUS_CALLED_PROCESS_ERROR
        stdout = e.stdout or ""
        stderr = e.stderr or ""
    except Exception as e:
        status = STATUS_EXCEPTION
        stderr = f"{type(e).__name__}: {e}"
    time_wall = time.perf_counter() - start

    bb_nodes, time_reported, lp_rows, lp_cols, lp_nonzeros = parse_concorde_stats(stdout, stderr)

    length = float("nan")
    valid = False
    tour0: Optional[list[int]] = None
    if sol_path.exists():
        route = read_solution_file(sol_path)
        norm = _normalize_tour_indices(route, n=n)
        if norm is not None:
            length = float(_tour_cost_int(mat_int, norm))
            valid = True
            tour0 = norm
        else:
            status = STATUS_BAD_TOUR
    else:
        if status == STATUS_OK:
            status = STATUS_NO_SOLUTION_FILE

    cleanup_concorde_files(scratch_dir, base_stub)
    # Our `scale_factor` is chosen to map from the (possibly scaled) training matrix
    # back to meaningful units (e.g. seconds), so the integer weights are already in
    # those units. Do not divide by scale_factor here.

    return SolveResult(
        valid=valid,
        status=int(status),
        length=float(length),
        time_wall=float(time_wall),
        time_reported=float(time_reported),
        bb_nodes=int(bb_nodes),
        lp_rows=int(lp_rows),
        lp_cols=int(lp_cols),
        lp_nonzeros=int(lp_nonzeros),
        tour=tour0,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect what-if node-removal dataset for cost-matrix TSP (Concorde).")
    p.add_argument("--run_dir", type=str, required=True, help="Run dir containing env.pkl + config.json (e.g. runs/matnet_seattle_big)")
    p.add_argument("--out_path", type=str, required=True, help="Output shard path (e.g. .../raw/shard_0000.pt)")
    p.add_argument("--num_instances", type=int, required=True, help="Total number of base instances across all shards.")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--concorde_timeout_sec", type=float, default=60.0)
    p.add_argument("--max_removed_nodes", type=int, default=None, help="Optional: only solve first K node removals per instance (debug).")
    p.add_argument("--assert_num_loc", type=int, default=None)
    p.add_argument("--log_every_sec", type=float, default=15.0)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    out_path = Path(args.out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        print(f"[collect] Output exists, skipping (use --overwrite): {out_path}")
        return

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not (run_dir / "env.pkl").exists():
        raise FileNotFoundError(f"Missing env.pkl in {run_dir}")
    if not (run_dir / "config.json").exists():
        raise FileNotFoundError(f"Missing config.json in {run_dir}")

    # Ensure repo imports work if run from elsewhere.
    repo_root = _repo_root()
    if str(repo_root) not in os.sys.path:
        os.sys.path.append(str(repo_root))

    config = json.loads((run_dir / "config.json").read_text())
    cost_scale = float(config.get("cost_scale", 1.0) or 1.0)
    if not (cost_scale > 0):
        cost_scale = 1.0

    with open(run_dir / "env.pkl", "rb") as fp:
        env = pickle.load(fp)
    patch_env_specs(env)

    start_idx, end_idx = shard_bounds(int(args.num_instances), int(args.num_shards), int(args.shard_idx))
    shard_size = end_idx - start_idx

    shard_seed = int(args.seed) + int(args.shard_idx)
    set_seed(shard_seed)

    if shard_size == 0:
        payload = {
            "meta": {
                "run_dir": str(run_dir),
                "seed": int(args.seed),
                "shard_seed": int(shard_seed),
                "num_instances_total": int(args.num_instances),
                "num_shards": int(args.num_shards),
                "shard_idx": int(args.shard_idx),
                "global_instance_start": int(start_idx),
                "global_instance_end": int(end_idx),
                "num_loc": 0,
                "concorde_timeout_sec": float(args.concorde_timeout_sec) if args.concorde_timeout_sec else None,
                "cost_scale": float(cost_scale),
                "created_at_unix": float(time.time()),
            }
        }
        torch.save(payload, out_path)
        print(f"[collect] Wrote empty shard: {out_path} (no instances)")
        return

    # Sample fixed instances (and lon/lat coords if available) using generator.sample_with_meta when present.
    cm = None
    coords = None
    idxs = None
    if hasattr(env, "generator") and hasattr(env.generator, "sample_with_meta"):
        cm, idxs, coords = env.generator.sample_with_meta(int(shard_size))
        cm = cm.to(torch.float32)
        td = env.reset(td=TensorDict({"cost_matrix": cm}, batch_size=[int(shard_size)])).to("cpu")
    else:
        td = env.reset(batch_size=[int(shard_size)]).to("cpu")
        if "cost_matrix" not in td:
            raise KeyError(f"Env reset did not provide cost_matrix; keys={list(td.keys(True, True))}")
        cm = td["cost_matrix"].detach().cpu().to(torch.float32)

    if cm.ndim != 3 or cm.shape[1] != cm.shape[2]:
        raise ValueError(f"Expected cost_matrix [B,n,n], got {tuple(cm.shape)}")
    B, n, _ = cm.shape

    if args.assert_num_loc is not None and int(args.assert_num_loc) != int(n):
        raise ValueError(f"assert_num_loc failed: expected {args.assert_num_loc}, got {n}")

    if coords is None:
        # Fallback coords for downstream tooling (not used for costs).
        theta = torch.linspace(0, 2 * math.pi, steps=n + 1)[:-1]
        circle = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1).to(torch.float32)
        coords = circle.unsqueeze(0).expand(B, n, 2).contiguous().numpy()
    locs = torch.from_numpy(np.asarray(coords, dtype=np.float32))

    base_length = torch.full((B,), float("nan"), dtype=torch.float32)
    base_time_wall = torch.full((B,), -1.0, dtype=torch.float32)
    base_time_reported = torch.full((B,), -1.0, dtype=torch.float32)
    base_bb_nodes = torch.full((B,), -1, dtype=torch.int64)
    base_lp_rows = torch.full((B,), -1, dtype=torch.int64)
    base_lp_cols = torch.full((B,), -1, dtype=torch.int64)
    base_lp_nonzeros = torch.full((B,), -1, dtype=torch.int64)
    base_status = torch.full((B,), -1, dtype=torch.int16)
    valid_base = torch.zeros((B,), dtype=torch.bool)

    minus_length = torch.full((B, n), float("nan"), dtype=torch.float32)
    minus_time_wall = torch.full((B, n), -1.0, dtype=torch.float32)
    minus_time_reported = torch.full((B, n), -1.0, dtype=torch.float32)
    minus_bb_nodes = torch.full((B, n), -1, dtype=torch.int64)
    minus_lp_rows = torch.full((B, n), -1, dtype=torch.int64)
    minus_lp_cols = torch.full((B, n), -1, dtype=torch.int64)
    minus_lp_nonzeros = torch.full((B, n), -1, dtype=torch.int64)
    minus_status = torch.full((B, n), -1, dtype=torch.int16)
    valid_minus = torch.zeros((B, n), dtype=torch.bool)

    tmp_root = out_path.parent
    scratch_dir = Path(tempfile.mkdtemp(dir=str(tmp_root), prefix=f"whatif_costmat_shard{args.shard_idx:04d}_"))

    max_removed_nodes = None
    if args.max_removed_nodes is not None:
        if int(args.max_removed_nodes) <= 0:
            raise ValueError(f"--max_removed_nodes must be >= 1, got {args.max_removed_nodes}")
        max_removed_nodes = min(int(args.max_removed_nodes), int(n))

    shard_start_t = time.perf_counter()
    last_log_t = shard_start_t
    solves_done = 0
    solves_per_instance = 1 + (n if max_removed_nodes is None else max_removed_nodes)
    total_solves = int(B) * int(solves_per_instance)

    def _format_eta(seconds: float) -> str:
        if not math.isfinite(seconds) or seconds < 0:
            return "?"
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            return f"{seconds/60:.1f}m"
        return f"{seconds/3600:.2f}h"

    def _maybe_log(j: int, global_id: int, phase: str, node_i: Optional[int]) -> None:
        nonlocal last_log_t
        now = time.perf_counter()
        if args.log_every_sec is not None and args.log_every_sec > 0 and (now - last_log_t) < float(args.log_every_sec):
            return
        last_log_t = now
        elapsed = now - shard_start_t
        rate = solves_done / elapsed if elapsed > 0 else float("nan")
        rem = max(0, total_solves - solves_done)
        eta = rem / rate if rate and rate > 0 else float("nan")
        node_str = "" if node_i is None else f" node={node_i}"
        print(
            f"[collect] shard {args.shard_idx}/{args.num_shards} inst {j}/{B} (global={global_id}) {phase}{node_str} "
            f"| solves {solves_done}/{total_solves} | {rate:.2f} solves/s | eta { _format_eta(eta) }"
        )

    try:
        for j in range(B):
            global_id = int(start_idx + j)
            base_stub = f"inst{j:05d}_base"
            _maybe_log(j, global_id, "base", None)
            res = solve_with_concorde_full_matrix(
                cm[j],
                scratch_dir=scratch_dir,
                base_stub=base_stub,
                scale_factor=cost_scale,
                timeout_sec=float(args.concorde_timeout_sec) if args.concorde_timeout_sec else None,
            )
            solves_done += 1

            base_status[j] = int(res.status)
            base_length[j] = float(res.length) if res.valid else float("nan")
            base_time_wall[j] = float(res.time_wall)
            base_time_reported[j] = float(res.time_reported)
            base_bb_nodes[j] = int(res.bb_nodes)
            base_lp_rows[j] = int(res.lp_rows)
            base_lp_cols[j] = int(res.lp_cols)
            base_lp_nonzeros[j] = int(res.lp_nonzeros)
            valid_base[j] = bool(res.valid)

            if not res.valid:
                continue

            # Solve all removals (or first K removals if debug).
            nodes_to_remove = range(n if max_removed_nodes is None else max_removed_nodes)
            for i in nodes_to_remove:
                _maybe_log(j, global_id, "minus", int(i))
                keep = [k for k in range(n) if k != i]
                sub = cm[j][keep][:, keep]
                res_m = solve_with_concorde_full_matrix(
                    sub,
                    scratch_dir=scratch_dir,
                    base_stub=f"inst{j:05d}_minus{i:03d}",
                    scale_factor=cost_scale,
                    timeout_sec=float(args.concorde_timeout_sec) if args.concorde_timeout_sec else None,
                )
                solves_done += 1

                minus_status[j, i] = int(res_m.status)
                minus_length[j, i] = float(res_m.length) if res_m.valid else float("nan")
                minus_time_wall[j, i] = float(res_m.time_wall)
                minus_time_reported[j, i] = float(res_m.time_reported)
                minus_bb_nodes[j, i] = int(res_m.bb_nodes)
                minus_lp_rows[j, i] = int(res_m.lp_rows)
                minus_lp_cols[j, i] = int(res_m.lp_cols)
                minus_lp_nonzeros[j, i] = int(res_m.lp_nonzeros)
                valid_minus[j, i] = bool(res_m.valid)

    finally:
        try:
            shutil.rmtree(scratch_dir, ignore_errors=True)
        except Exception:
            pass

    # Deltas
    base_rep = base_length.unsqueeze(1).expand(B, n)
    delta_length_pct = torch.full((B, n), float("nan"), dtype=torch.float32)
    denom = base_rep.clamp_min(1e-9)
    delta_length_pct = 100.0 * (base_rep - minus_length) / denom

    base_time_rep = base_time_wall.unsqueeze(1).expand(B, n)
    delta_time_pct = torch.full((B, n), float("nan"), dtype=torch.float32)
    denom_t = base_time_rep.clamp_min(1e-9)
    delta_time_pct = 100.0 * (base_time_rep - minus_time_wall) / denom_t

    payload: Dict[str, Any] = {
        "locs": locs,
        "cost_matrix": cm,
        "base_length": base_length,
        "base_time_wall": base_time_wall,
        "base_time_reported": base_time_reported,
        "base_bb_nodes": base_bb_nodes,
        "base_lp_rows": base_lp_rows,
        "base_lp_cols": base_lp_cols,
        "base_lp_nonzeros": base_lp_nonzeros,
        "base_status": base_status,
        "valid_base": valid_base,
        "minus_length": minus_length,
        "minus_time_wall": minus_time_wall,
        "minus_time_reported": minus_time_reported,
        "minus_bb_nodes": minus_bb_nodes,
        "minus_lp_rows": minus_lp_rows,
        "minus_lp_cols": minus_lp_cols,
        "minus_lp_nonzeros": minus_lp_nonzeros,
        "minus_status": minus_status,
        "valid_minus": valid_minus,
        "delta_length_pct": delta_length_pct,
        "delta_time_pct": delta_time_pct,
        "meta": {
            "run_dir": str(run_dir),
            "seed": int(args.seed),
            "shard_seed": int(shard_seed),
            "num_instances_total": int(args.num_instances),
            "num_shards": int(args.num_shards),
            "shard_idx": int(args.shard_idx),
            "global_instance_start": int(start_idx),
            "global_instance_end": int(end_idx),
            "num_loc": int(n),
            "concorde_timeout_sec": float(args.concorde_timeout_sec) if args.concorde_timeout_sec else None,
            "cost_scale": float(cost_scale),
            "pool_dir": config.get("pool_dir"),
            "tsplib_path": config.get("tsplib_path"),
            "created_at_unix": float(time.time()),
        },
    }
    if idxs is not None:
        payload["pool_indices"] = torch.from_numpy(np.asarray(idxs, dtype=np.int64))

    torch.save(payload, out_path)
    print(f"[collect] Wrote shard: {out_path} (instances {start_idx}..{end_idx-1}, n={n})")


if __name__ == "__main__":
    main()
