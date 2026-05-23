#!/usr/bin/env python3
"""Analysis of the 128^3 slip-wall blast run with nabla^6 FD hyperdissipation.

Two figures:
  figs/blast_128_dissipation.png  -- KE, eps_sol, eps_dil vs time
  figs/blast_128_spectra.png       -- E(k) at four selected times, with
                                      k^{-5/3} reference and compensated
                                      view on a second axis.

Reads from out_blast_128_slip_hyper6_fd/.
"""

import csv
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
RUN_DIR = run_dir("out_blast_128_slip_hyper6_fd")
STATS_PATH = RUN_DIR / "blast_128_slip_hyper6_fd_stats.csv"
SPECTRA_PATH = RUN_DIR / "blast_128_slip_hyper6_fd_spectra.h5"
FIGS_DIR = ROOT / "figs"


def load_stats():
    cols = {}
    with open(STATS_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return {k: np.array(v) for k, v in cols.items()}


def load_spectra():
    with h5py.File(SPECTRA_PATH, "r") as f:
        step_groups = sorted(g for g in f.keys() if g.startswith("step_"))
        times = np.array([float(f[g]["time"][0]) for g in step_groups])
        k = np.array(f[step_groups[0]]["k"])
        E_total = np.stack([np.array(f[g]["E_total"]) for g in step_groups])
        E_sol   = np.stack([np.array(f[g]["E_sol"  ]) for g in step_groups])
        E_dil   = np.stack([np.array(f[g]["E_dil"  ]) for g in step_groups])
    return times, k, E_total, E_sol, E_dil


def plot_dissipation():
    s = load_stats()
    t = s["time"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.plot(t, s["KE"], lw=1.6, label="KE")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("kinetic energy")
    ax.set_title("KE(t)")
    ax.grid(True, ls=":", lw=0.5, alpha=0.5)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.semilogy(t, s["eps_total"], "-", color="k", lw=1.4,
                label=r"$\varepsilon_{\rm total}$")
    ax.semilogy(t, s["eps_sol"], "-", color="tab:blue", lw=1.4,
                label=r"$\varepsilon_{\rm sol}$ (solenoidal)")
    ax.semilogy(t, s["eps_dil"], "-", color="tab:red", lw=1.4,
                label=r"$\varepsilon_{\rm dil}$ (dilatational)")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\varepsilon$")
    ax.set_title(r"Dissipation rate components")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
    ax.legend(frameon=False)

    fig.suptitle(r"128$^3$ slip-wall blast, $\nabla^6$ FD: energy diagnostics",
                 y=1.02)
    out = FIGS_DIR / "blast_128_dissipation.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


def plot_spectra():
    times, k, E_total, E_sol, E_dil = load_spectra()
    s = load_stats()
    t_peak = s["time"][int(np.argmax(s["eps_total"]))]

    # Snapshot times: peak, post-peak, mid, late.
    snap_targets = [t_peak,
                    min(t_peak + 0.05, times[-1]),
                    0.25,
                    times[-1]]

    # Time-averaging window: post-shock turbulent state.
    # eps_total has settled to ~0.5 baseline by t~0.1; average over [0.15, t_end].
    avg_mask = (times >= 0.15) & (times <= times[-1])
    E_avg = E_total[avg_mask].mean(axis=0)
    avg_lo, avg_hi = times[avg_mask][0], times[avg_mask][-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # k_max for the inertial-range reference is the largest k present.
    k_top = float(k.max())

    amp_anchor = None
    for t_target in snap_targets:
        idx = int(np.argmin(np.abs(times - t_target)))
        t_used = times[idx]
        Ek = E_total[idx]
        mask = (k > 0) & (Ek > 0)
        line, = axes[0].loglog(k[mask], Ek[mask], lw=1.2, alpha=0.55,
                               label=f"t={t_used:.3f}")
        if amp_anchor is None:
            idx_k = int(np.argmin(np.abs(k - 5.0)))
            amp_anchor = Ek[idx_k] * (5.0 ** (5.0 / 3.0))
        axes[1].loglog(k[mask], Ek[mask] * k[mask] ** (5.0 / 3.0),
                       lw=1.2, alpha=0.55, color=line.get_color(),
                       label=f"t={t_used:.3f}")

    # Overlay time-averaged spectrum (post-shock window).
    mask_avg = (k > 0) & (E_avg > 0)
    axes[0].loglog(k[mask_avg], E_avg[mask_avg], "k-", lw=2.0,
                   label=rf"$\langle E\rangle_{{t\in[{avg_lo:.2f},{avg_hi:.2f}]}}$")
    axes[1].loglog(k[mask_avg], E_avg[mask_avg] * k[mask_avg] ** (5.0 / 3.0),
                   "k-", lw=2.0,
                   label=rf"$\langle E\rangle_{{t\in[{avg_lo:.2f},{avg_hi:.2f}]}}$")

    # Re-anchor reference line to the time-averaged spectrum at k=5.
    idx_k5 = int(np.argmin(np.abs(k - 5.0)))
    amp_anchor = E_avg[idx_k5] * (5.0 ** (5.0 / 3.0))
    kref = np.logspace(np.log10(2.0), np.log10(min(k_top, 100.0)), 60)
    axes[0].loglog(kref, amp_anchor * kref ** (-5.0 / 3.0),
                   "r--", lw=1.2, label=r"$k^{-5/3}$ ref")
    axes[1].axhline(amp_anchor, color="r", ls="--", lw=1.2,
                    label=r"$k^{-5/3}$ flat here")

    for ax in axes:
        ax.set_xlabel(r"$k$")
        ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        ax.set_xlim(1.5, min(k_top, 250))
    axes[0].set_ylabel(r"$E(k)$")
    axes[0].set_title(r"Spectra $E(k)$ (snapshots + time avg)")
    axes[0].set_ylim(1e-9, 1e1)
    axes[1].set_ylabel(r"$E(k)\, k^{5/3}$")
    axes[1].set_title(r"Compensated $E(k)\, k^{5/3}$")
    axes[1].set_ylim(1e-4, 1e1)

    fig.suptitle(r"128$^3$ slip-wall blast, $\nabla^6$ FD: spectra at selected times",
                 y=1.02)
    out = FIGS_DIR / "blast_128_spectra.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


def main():
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    if not STATS_PATH.exists():
        print(f"missing: {STATS_PATH}", file=sys.stderr)
        return 1
    plot_dissipation()
    if SPECTRA_PATH.exists():
        plot_spectra()
    else:
        print(f"missing: {SPECTRA_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
