#!/usr/bin/env python3
"""Compressibility-regime diagnostics for the 128^3 slip-wall blast.

Tests the hypothesis from the paper that the post-blast Kolmogorov-like
inertial range arises because the chamber equilibrates to elevated pressure
-> elevated sound speed -> low turbulent Mach -> effectively low-compressibility
turbulence.

Plots p_mean(t), T_mean(t), c_mean(t), M_t(t), K_dil/K_sol(t), and
eps_dil/eps_sol(t) on one figure, with the initial ambient values marked.
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

# Use the longer-integration run when available -- it has the cleanest
# late-time behaviour.
LONG = RUNS / "out_blast_128_slip_hyper6_fd_long" / \
       "blast_128_slip_hyper6_fd_long_stats.csv"
SHORT = RUNS / "out_blast_128_slip_hyper6_fd" / \
        "blast_128_slip_hyper6_fd_stats.csv"


def load(stats_csv):
    cols = {}
    with open(stats_csv) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return {k: np.array(v) for k, v in cols.items()}


def main():
    src = LONG if LONG.exists() else SHORT
    s = load(src)
    t = s["time"]

    # Ambient (initial) reference values.
    rho0, T0 = 1.0, 1.0
    gamma, R = 1.4, 1.0
    p0 = rho0 * R * T0
    c0 = np.sqrt(gamma * R * T0)   # ~1.183

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    ax = axes[0, 0]
    ax.plot(t, s["p_mean"], lw=1.6)
    ax.axhline(p0, color="gray", ls=":", lw=1.0, label=f"ambient $p_0={p0:.2f}$")
    ax.set_ylabel(r"$\bar p$")
    ax.set_title(r"mean pressure $\bar p(t)$")
    ax.legend(frameon=False)
    ax.grid(True, ls=":", alpha=0.5)

    ax = axes[0, 1]
    ax.plot(t, s["T_mean"], lw=1.6)
    ax.axhline(T0, color="gray", ls=":", lw=1.0,
               label=f"ambient $T_0={T0:.2f}$")
    ax.set_ylabel(r"$\bar T$")
    ax.set_title(r"mean temperature $\bar T(t)$")
    ax.legend(frameon=False)
    ax.grid(True, ls=":", alpha=0.5)

    ax = axes[0, 2]
    ax.plot(t, s["c_mean"], lw=1.6)
    ax.axhline(c0, color="gray", ls=":", lw=1.0,
               label=f"ambient $c_0={c0:.2f}$")
    ax.set_ylabel(r"$\bar c$")
    ax.set_title(r"mean sound speed $\bar c(t)$")
    ax.legend(frameon=False)
    ax.grid(True, ls=":", alpha=0.5)

    ax = axes[1, 0]
    ax.semilogy(t, s["M_t"], lw=1.6, color="tab:purple")
    ax.axhline(1.0, color="gray", ls=":", lw=1.0,
               label=r"$M_t=1$ (sonic)")
    ax.axhline(0.3, color="gray", ls="--", lw=1.0,
               label=r"$M_t=0.3$ ('low-Mach' threshold)")
    ax.set_ylabel(r"$M_t$")
    ax.set_xlabel(r"$t$")
    ax.set_title(r"turbulent Mach $M_t = u_{\rm rms}/\bar c$")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.5)

    ax = axes[1, 1]
    K_ratio = np.maximum(s.get("K_dil", s.get("eps_dil")) /
                         np.maximum(s.get("K_sol", s.get("eps_sol")), 1e-30),
                         1e-30)
    # Prefer K_dil/K_sol from stats if column exists, else use eps ratio.
    if "K_dil" in s and "K_sol" in s:
        ratio = s["K_dil"] / np.maximum(s["K_sol"], 1e-30)
        label = r"$K_{\rm dil} / K_{\rm sol}$"
    else:
        ratio = s["eps_dil"] / np.maximum(s["eps_sol"], 1e-30)
        label = r"$\varepsilon_{\rm dil} / \varepsilon_{\rm sol}$"
    ax.semilogy(t, ratio, lw=1.6, color="tab:red")
    ax.axhline(1.0, color="gray", ls=":", lw=1.0, label="equal partition")
    ax.set_ylabel(label)
    ax.set_xlabel(r"$t$")
    ax.set_title("compressible vs solenoidal energy partition")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.5)

    ax = axes[1, 2]
    eps_ratio = s["eps_dil"] / np.maximum(s["eps_sol"], 1e-30)
    ax.semilogy(t, eps_ratio, lw=1.6, color="tab:olive")
    ax.axhline(1.0, color="gray", ls=":", lw=1.0)
    ax.set_ylabel(r"$\varepsilon_{\rm dil} / \varepsilon_{\rm sol}$")
    ax.set_xlabel(r"$t$")
    ax.set_title("dissipation partition")
    ax.grid(True, which="both", ls=":", alpha=0.5)

    fig.suptitle(
        "128$^3$ slip-wall blast: compressibility metrics over the run\n"
        r"hypothesis: elevated $\bar p$ raises $\bar c$, drives $M_t \ll 1$, "
        r"and lets Kolmogorov $k^{-5/3}$ form",
        y=1.02, fontsize=12)
    out = ROOT / "figs" / "blast_128_compressibility.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")

    # Also print key numbers at three time points.
    print("\nKey diagnostics over time (from", src.name, "):")
    print(f"  ambient: p={p0:.3f}, T={T0:.3f}, c={c0:.3f}")
    for t_check in [0.05, 0.10, 0.25, 0.50, t[-1]]:
        if t_check > t[-1]:
            continue
        i = int(np.argmin(np.abs(t - t_check)))
        line = (f"  t={t[i]:.3f}: p={s['p_mean'][i]:.3f} "
                f"(={s['p_mean'][i]/p0:.1f}x p0), "
                f"c={s['c_mean'][i]:.3f} (={s['c_mean'][i]/c0:.1f}x c0), "
                f"M_t={s['M_t'][i]:.3f}, "
                f"eps_dil/eps_sol={s['eps_dil'][i]/max(s['eps_sol'][i],1e-30):.2f}")
        print(line)


if __name__ == "__main__":
    main()
