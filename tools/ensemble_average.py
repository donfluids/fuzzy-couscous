"""
Ensemble-average several `<run>_stats.csv` files produced from runs of the
same config differing only in `ensemble_seed`. Addresses reviewer M6.

Aligns the realizations on a common time grid (linear interpolation), then
reports mean and standard error of every numeric column.

Usage:
    python tools/ensemble_average.py <stats_a.csv> <stats_b.csv> ... \
        --out ensemble_mean.csv [--plot column[,column,...]]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def common_time_grid(dfs: list[pd.DataFrame], n_points: int = 400) -> np.ndarray:
    t_min = max(df["time"].min() for df in dfs)
    t_max = min(df["time"].max() for df in dfs)
    if t_max <= t_min:
        raise ValueError("realizations have non-overlapping time windows")
    return np.linspace(t_min, t_max, n_points)


def interpolate(df: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    out = {"time": grid}
    for col in df.columns:
        if col == "time":
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            continue
        out[col] = np.interp(grid, df["time"].to_numpy(), df[col].to_numpy())
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plot", type=str, default=None,
                    help="comma-separated column names to plot mean +/- 1 SE")
    args = ap.parse_args()

    dfs = [pd.read_csv(p) for p in args.csvs]
    if len(dfs) < 2:
        print("warning: only one realization given; reporting it verbatim",
              file=sys.stderr)
    grid = common_time_grid(dfs) if len(dfs) > 1 else dfs[0]["time"].to_numpy()

    aligned = [interpolate(df, grid) for df in dfs]
    cols = [c for c in aligned[0].columns if c != "time"]
    stack = np.stack([df[cols].to_numpy() for df in aligned], axis=0)
    mean = stack.mean(axis=0)
    se = stack.std(axis=0, ddof=1) / np.sqrt(len(dfs)) if len(dfs) > 1 \
        else np.zeros_like(mean)

    out_cols = {"time": grid}
    for i, c in enumerate(cols):
        out_cols[c] = mean[:, i]
        out_cols[c + "_se"] = se[:, i]
    out_df = pd.DataFrame(out_cols)
    out_df.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(dfs)} realizations, {len(grid)} time points)")

    if args.plot:
        wanted = args.plot.split(",")
        fig, ax = plt.subplots(figsize=(7, 4))
        for c in wanted:
            if c not in cols:
                print(f"  skip plot column '{c}' (not in CSVs)", file=sys.stderr)
                continue
            i = cols.index(c)
            ax.plot(grid, mean[:, i], label=c)
            ax.fill_between(grid, mean[:, i] - se[:, i], mean[:, i] + se[:, i],
                            alpha=0.2)
        ax.set_xlabel("time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = args.out.with_suffix(".png")
        fig.savefig(plot_path, dpi=120)
        print(f"plot:  {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
