import argparse
import csv
import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def _latest_metrics_csv(run_dir: Path) -> Path:
    logs = run_dir / "logs"
    if not logs.exists():
        raise FileNotFoundError(f"Missing logs dir: {logs}")
    versions = []
    for p in logs.glob("version_*"):
        try:
            versions.append((int(p.name.split("_", 1)[1]), p))
        except Exception:
            continue
    if not versions:
        raise FileNotFoundError(f"No logs/version_*/ found under {logs}")
    _, latest = max(versions, key=lambda x: x[0])
    csv_path = latest / "metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metrics.csv: {csv_path}")
    return csv_path


def _load_baseline_cost(run_dir: Path, baseline_name: str) -> float:
    path = run_dir / baseline_name
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline file: {path}")
    baseline = pickle.load(open(path, "rb"))
    rewards = baseline.get("rewards")
    if not isinstance(rewards, list) or not rewards:
        raise ValueError(f"Unexpected baseline format in {path}: expected dict with non-empty list rewards")
    r0 = rewards[0]
    if not hasattr(r0, "mean"):
        raise ValueError(f"Unexpected rewards[0] type in {path}: {type(r0)}")
    return float((-r0).mean().item())


def _read_val_costs(metrics_csv: Path, step: int) -> tuple[np.ndarray, np.ndarray]:
    by_epoch: dict[int, float] = {}
    with metrics_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "epoch" not in reader.fieldnames or "val/reward" not in reader.fieldnames:
            raise ValueError(f"{metrics_csv} must contain columns 'epoch' and 'val/reward'")
        for row in reader:
            e = row.get("epoch")
            vr = row.get("val/reward")
            if e is None or vr is None or vr == "" or vr.lower() == "nan":
                continue
            try:
                epoch = int(float(e))
                val_reward = float(vr)
            except Exception:
                continue
            by_epoch[epoch] = val_reward

    if not by_epoch:
        raise ValueError(f"No val/reward entries found in {metrics_csv}")

    epochs = np.array(sorted(by_epoch.keys()), dtype=int)
    epochs = epochs[epochs % int(step) == 0]
    vals = np.array([by_epoch[int(e)] for e in epochs], dtype=float)
    costs = -vals
    return epochs, costs


def _double_exp_fixed_asymptote(x: np.ndarray, a: float, b: float, c: float, d: float, *, asymptote: float) -> np.ndarray:
    return asymptote + a * np.exp(-b * x) + c * np.exp(-d * x)


def main(args: argparse.Namespace) -> None:
    run_dir = Path("runs") / args.run_name
    metrics_csv = _latest_metrics_csv(run_dir)
    epochs, costs = _read_val_costs(metrics_csv, step=int(args.step))

    baseline_cost = None
    if args.baseline is not None:
        baseline_cost = _load_baseline_cost(run_dir, args.baseline)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, costs, marker="o", markersize=2, label="Avg Distance per Epoch")

    if baseline_cost is not None:
        plt.axhline(y=baseline_cost, color="red", linestyle="--", label="Baseline (Optimal)")

        # Optional double-exponential fit with fixed asymptote at the baseline.
        if len(epochs) >= 6:
            x = epochs.astype(np.float64)
            y = costs.astype(np.float64)
            # Initial guess: two time scales.
            a0 = float(y[0] - baseline_cost)
            c0 = float(0.5 * (y[0] - baseline_cost))
            p0 = [a0, 0.05, c0, 0.005]
            bounds = ([-np.inf, 0.0, -np.inf, 0.0], [np.inf, np.inf, np.inf, np.inf])
            try:
                popt, _pcov = curve_fit(
                    lambda xx, a, b, c, d: _double_exp_fixed_asymptote(xx, a, b, c, d, asymptote=baseline_cost),
                    x,
                    y,
                    p0=p0,
                    bounds=bounds,
                    maxfev=50_000,
                )
                x_fit = np.linspace(float(x.min()), float(x.max()), 250)
                y_fit = _double_exp_fixed_asymptote(x_fit, *popt, asymptote=baseline_cost)
                if np.isfinite(y_fit).all() and not math.isclose(float(y_fit.std()), 0.0):
                    plt.plot(x_fit, y_fit, color="green", linestyle=":", label="Double Exp Fit (Fixed Asymptote)")
            except Exception:
                pass

    plt.xlabel("Epoch")
    plt.ylabel("Distance")
    plt.title(f"Average Distance per Training Epoch vs. Baseline (every {int(args.step)} epochs)")
    plt.legend()

    y_min = float(np.min(costs))
    y_max = float(np.max(costs))
    if baseline_cost is not None:
        y_min = min(y_min, float(baseline_cost))
    pad = 0.02 * (y_max - y_min) if y_max > y_min else 1.0
    plt.ylim(y_min - pad, y_max + pad)

    out = run_dir / "train_plot.png"
    plt.savefig(out)
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--step", type=int, default=5)
    p.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Baseline pickle filename in the run dir (e.g. baseline_concorde.pkl). If unset, no baseline/fit is drawn.",
    )
    main(p.parse_args())

