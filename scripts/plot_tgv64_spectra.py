#!/usr/bin/env python3
"""TGV 64^3 spectra comparison: nabla^4 FD baseline vs nabla^6 FD vs nabla^6 spectral.

Reads the spectra HDF5 files written by blast_les / blast_les_mpi (one Group
per dump step, each holding E_total[k], E_sol[k], E_dil[k], k[k], time[1]),
finds the time of peak total dissipation in each, and plots E_total(k) on a
log-log axis with a k^{-5/3} reference line. Output: figs/tgv64_spectra.png.
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent

RUNS = [
    # (label, out_dir, spectra_filename, stats_filename, color)
    (
        r"$\nabla^4$ FD (baseline)",
        "out_tgv64_hyper2",
        "tgv_re1600_64_hyper2_spectra.h5",
        "tgv_re1600_64_hyper2_stats.csv",
        "tab:blue",
    ),
    (
        r"$\nabla^6$ FD",
        "out_tgv64_hyper6_fd",
        "tgv_re1600_64_hyper6_fd_spectra.h5",
        "tgv_re1600_64_hyper6_fd_stats.csv",
        "tab:orange",
    ),
    (
        r"$\nabla^6$ spectral",
        "out_tgv64_hyper6_spectral",
        "tgv_re1600_64_hyper6_spectral_spectra.h5",
        "tgv_re1600_64_hyper6_spectral_stats.csv",
        "tab:green",
    ),
]


def load_spectra(h5path):
    """Return arrays times[N], k[Nk], E_total[N, Nk]."""
    with h5py.File(h5path, "r") as f:
        step_groups = sorted(g for g in f.keys() if g.startswith("step_"))
        times = np.array([float(f[g]["time"][0]) for g in step_groups])
        k = np.array(f[step_groups[0]]["k"])
        E = np.stack([np.array(f[g]["E_total"]) for g in step_groups])
    return times, k, E


def peak_dissipation_time(stats_csv):
    """Read CSV stats and return the time at which eps_total is maximum."""
    import csv
    times, eps = [], []
    with open(stats_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time"]))
            eps.append(float(row["eps_total"]))
    times = np.array(times)
    eps = np.array(eps)
    return float(times[int(np.argmax(eps))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", type=float, default=None,
                    help="time slice to plot; default = peak-eps per run")
    ap.add_argument("--out", default=str(ROOT / "figs" / "tgv64_spectra.png"))
    args = ap.parse_args()

    def plot_one(ax, t_chooser, title):
        amp_anchor = None
        for label, out_dir, spec_name, stats_name, color in RUNS:
            spec_path = ROOT / "runs" / out_dir / spec_name
            stats_path = ROOT / "runs" / out_dir / stats_name
            if not spec_path.exists():
                print(f"missing: {spec_path}", file=sys.stderr)
                continue
            times, k, E = load_spectra(spec_path)
            t_target = t_chooser(stats_path)
            idx = int(np.argmin(np.abs(times - t_target)))
            t_used = times[idx]
            Ek = E[idx]
            mask = (k > 0) & (Ek > 0)
            ax.loglog(k[mask], Ek[mask], "-", color=color, lw=1.8,
                      label=f"{label} (t={t_used:.2f})")
            if amp_anchor is None:
                idx_k = int(np.argmin(np.abs(k - 5.0)))
                amp_anchor = Ek[idx_k] * (5.0 ** (5.0 / 3.0))
        if amp_anchor is None:
            amp_anchor = 1e-3
        kref = np.logspace(np.log10(2.0), np.log10(30.0), 50)
        ax.loglog(kref, amp_anchor * kref ** (-5.0 / 3.0),
                  "k--", lw=1.2, label=r"$k^{-5/3}$")
        ax.set_xlabel(r"$k$")
        ax.set_ylabel(r"$E(k)$")
        ax.set_title(title)
        ax.set_xlim(1.5, 35)
        ax.set_ylim(1e-7, 2e-1)
        ax.legend(frameon=False, fontsize=9, loc="lower left")
        ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    plot_one(axes[0], peak_dissipation_time,
             r"each run at its own peak dissipation")
    plot_one(axes[1], lambda _: 9.0,
             r"all runs at common $t=9$")
    fig.suptitle(r"TGV Re=1600, $64^3$: $E(k)$ comparison", y=1.02)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
