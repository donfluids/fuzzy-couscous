#!/usr/bin/env python3
"""Compensated spectra E(k)*k^{5/3} for 64^3 and 128^3 TGV runs.

A k^{-5/3} inertial range shows up as a horizontal plateau on these axes,
making the inertial-range extent obvious by eye. Output:
figs/tgv_compensated_spectra.png.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


import sys
from pathlib import Path
_PP = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").is_file())
for _d in (_PP, _PP / "tools", _PP / "scripts"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
from paths import REPO_ROOT as ROOT, RUNS, DATA, FIGS, run_dir  # noqa: E402


def load_spectra(h5path):
    with h5py.File(h5path, "r") as f:
        step_groups = sorted(g for g in f.keys() if g.startswith("step_"))
        times = np.array([float(f[g]["time"][0]) for g in step_groups])
        k = np.array(f[step_groups[0]]["k"])
        E = np.stack([np.array(f[g]["E_total"]) for g in step_groups])
    return times, k, E


def peak_dissipation_time(stats_csv):
    import csv
    t, eps = [], []
    with open(stats_csv) as f:
        for row in csv.DictReader(f):
            t.append(float(row["time"]))
            eps.append(float(row["eps_total"]))
    t = np.array(t); eps = np.array(eps)
    return float(t[int(np.argmax(eps))])


def runs_for(N):
    return [
        (r"$\nabla^4$ FD (baseline)",
         f"out_tgv{N}_hyper2",
         f"tgv_re1600_{N}_hyper2_spectra.h5",
         f"tgv_re1600_{N}_hyper2_stats.csv",
         "tab:blue"),
        (r"$\nabla^6$ FD",
         f"out_tgv{N}_hyper6_fd",
         f"tgv_re1600_{N}_hyper6_fd_spectra.h5",
         f"tgv_re1600_{N}_hyper6_fd_stats.csv",
         "tab:orange"),
        (r"$\nabla^6$ spectral",
         f"out_tgv{N}_hyper6_spectral",
         f"tgv_re1600_{N}_hyper6_spectral_spectra.h5",
         f"tgv_re1600_{N}_hyper6_spectral_stats.csv",
         "tab:green"),
    ]


def plot_panel(ax, runs, title, k_xmax):
    for label, out_dir, spec_name, stats_name, color in runs:
        spec_path = run_dir(out_dir) / spec_name
        stats_path = run_dir(out_dir) / stats_name
        if not spec_path.exists():
            print(f"missing: {spec_path}", file=sys.stderr)
            continue
        times, k, E = load_spectra(spec_path)
        t_target = peak_dissipation_time(stats_path)
        idx = int(np.argmin(np.abs(times - t_target)))
        Ek = E[idx]
        mask = (k > 0) & (Ek > 0)
        ax.loglog(k[mask], Ek[mask] * k[mask] ** (5.0 / 3.0), "-",
                  color=color, lw=1.8,
                  label=f"{label} (t={times[idx]:.2f})")
    # Horizontal -5/3 reference would be a single constant; pick a
    # representative value from k=5 region of the first available run.
    for _, out_dir, spec_name, stats_name, _ in runs:
        spec_path = run_dir(out_dir) / spec_name
        if spec_path.exists():
            times, k, E = load_spectra(spec_path)
            t_target = peak_dissipation_time(run_dir(out_dir) / stats_name)
            idx = int(np.argmin(np.abs(times - t_target)))
            idx_k = int(np.argmin(np.abs(k - 5.0)))
            ref = E[idx, idx_k] * (5.0 ** (5.0 / 3.0))
            ax.axhline(ref, color="k", ls="--", lw=1.2,
                       label=r"$k^{-5/3}$ (flat here)")
            break
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E(k)\, k^{5/3}$")
    ax.set_title(title)
    ax.set_xlim(1.5, k_xmax)
    ax.set_ylim(1e-4, 3e-1)
    ax.legend(frameon=False, fontsize=9, loc="lower center")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    plot_panel(axes[0], runs_for(64),
               r"$64^3$ (compensated, peak dissipation)", 35)
    plot_panel(axes[1], runs_for(128),
               r"$128^3$ (compensated, peak dissipation)", 70)
    fig.suptitle(
        r"Compensated TGV spectra: a flat plateau on these axes is $k^{-5/3}$",
        y=1.02)
    out = ROOT / "figs" / "tgv_compensated_spectra.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
