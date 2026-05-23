"""
Fit power-law decay tke(t) ~ A * (t - t0) ** -n to the stats CSV produced by
the blast_les driver. Bootstrap confidence intervals on the exponent.

Usage:
    python tools/fit_decay.py <stats.csv> [--t-min T] [--t-max T] [--col tke]

Outputs a one-line summary and (optionally) a matplotlib figure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def fit_power_law(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Linear fit in log-log: log y = log A + (-n) * log t.

    Returns (A, n, residual_std).
    """
    mask = (t > 0) & (y > 0)
    if mask.sum() < 4:
        return float("nan"), float("nan"), float("nan")
    lt = np.log(t[mask])
    ly = np.log(y[mask])
    slope, intercept = np.polyfit(lt, ly, 1)
    pred = intercept + slope * lt
    residual_std = float(np.std(ly - pred))
    return float(np.exp(intercept)), float(-slope), residual_std


def bootstrap_ci(
    t: np.ndarray, y: np.ndarray, n_boot: int = 1000, rng: np.random.Generator | None = None
) -> tuple[float, float]:
    """95% CI on the exponent via residual bootstrap."""
    if rng is None:
        rng = np.random.default_rng(0)
    mask = (t > 0) & (y > 0)
    lt = np.log(t[mask])
    ly = np.log(y[mask])
    N = len(lt)
    if N < 8:
        return float("nan"), float("nan")
    slope0, intercept0 = np.polyfit(lt, ly, 1)
    resid = ly - (intercept0 + slope0 * lt)
    exponents = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.integers(0, N, N)
        ly_boot = (intercept0 + slope0 * lt) + resid[sample]
        s, _ = np.polyfit(lt, ly_boot, 1)
        exponents[b] = -s
    lo, hi = np.percentile(exponents, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--col", default="tke", help="column to fit (default: tke)")
    ap.add_argument("--t-min", type=float, default=None)
    ap.add_argument("--t-max", type=float, default=None)
    ap.add_argument("--plot", type=Path, default=None,
                    help="write fit figure to this PNG path")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, skipinitialspace=True)
    if args.col not in df.columns:
        print(f"error: column {args.col} not in {list(df.columns)}", file=sys.stderr)
        return 1

    t = df["time"].to_numpy()
    y = df[args.col].to_numpy()

    if args.t_min is not None:
        mask = t >= args.t_min
        t, y = t[mask], y[mask]
    if args.t_max is not None:
        mask = t <= args.t_max
        t, y = t[mask], y[mask]

    A, n, resid = fit_power_law(t, y)
    lo, hi = bootstrap_ci(t, y, n_boot=2000)

    print(f"file:    {args.csv}")
    print(f"column:  {args.col}")
    print(f"window:  t in [{t.min():.4e}, {t.max():.4e}], {len(t)} samples")
    print(f"fit:     {args.col} = {A:.4e} * t^(-{n:.4f})")
    print(f"95% CI:  n in [{lo:.4f}, {hi:.4f}]")
    print(f"residual log-std: {resid:.4f}")

    if args.plot is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(t, y, "k.", ms=3, label="data")
        ts = np.linspace(max(t.min(), 1e-12), t.max(), 200)
        ax.loglog(ts, A * ts ** (-n), "r-", lw=1,
                  label=f"$\\propto t^{{-{n:.2f}}}$ (95% CI [{lo:.2f}, {hi:.2f}])")
        ax.set_xlabel("time")
        ax.set_ylabel(args.col)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"figure: {args.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
