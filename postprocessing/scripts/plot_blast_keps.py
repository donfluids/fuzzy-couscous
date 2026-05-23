#!/usr/bin/env python3
"""Test whether the post-equilibration blast turbulence is amenable to k-eps modeling.

Two empirical checks, both from the long 128^3 run:

  1. dK/dt vs -eps balance: in a homogeneous decay or local k-eps
     regime, dK_sol/dt = -eps_sol (production = 0 in a closed box).
     We compute the numerical derivative and overplot -eps_sol.

  2. Effective C_eps2: for pure decay,
        K(t) ~ (t - t0)^{-n},  n = 1/(C_eps2 - 1).
     Fit a power law to K_sol(t) in the late stage and back out C_eps2.
     Standard k-eps uses C_eps2 = 1.92 -> n = 1.087.

  Also overplots K_sol vs K_dil to show what fraction the model would
  be capturing.
"""

import csv
import re
from pathlib import Path

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
STATS = run_dir("out_blast_128_slip_hyper6_fd_long") / \
        "blast_128_slip_hyper6_fd_long_stats.csv"
LOG_FILE = DATA / "run_blast_128_slip_hyper6_fd_long.log"


def parse_kdksol_from_log(path):
    """Return (times, K_dil/K_sol ratios) parsed from the run log."""
    pat = re.compile(r"t=([\d.eE+-]+).*K_dil/K_sol=([\d.eE+-]+)")
    ts, rs = [], []
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                ts.append(float(m.group(1)))
                rs.append(float(m.group(2)))
    return np.array(ts), np.array(rs)


def load_stats(path):
    cols = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return {k: np.array(v) for k, v in cols.items()}


def main():
    s = load_stats(STATS)
    t = s["time"]
    KE = s["KE"]
    eps_sol = s["eps_sol"]
    eps_dil = s["eps_dil"]
    eps_tot = s["eps_total"]

    # Parse K_dil/K_sol ratio from log (stats CSV omits it).
    t_log, ratio_log = parse_kdksol_from_log(LOG_FILE)
    # Interpolate ratio onto stats times.
    ratio = np.interp(t, t_log, ratio_log)
    # K_sol = K_total / (1 + K_dil/K_sol); K_dil = K_total - K_sol.
    K_sol = KE / (1.0 + ratio)
    K_dil = KE - K_sol

    # Heavy smoothing -- the chamber's acoustic ringing dominates the
    # instantaneous derivative; we need a wide window to see the decay
    # envelope.
    win = 25
    K_sol_s = np.convolve(K_sol, np.ones(win)/win, mode="same")
    eps_sol_s = np.convolve(eps_sol, np.ones(win)/win, mode="same")
    eps_tot_s = np.convolve(eps_tot, np.ones(win)/win, mode="same")

    dKdt = np.gradient(K_sol_s, t)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: K_sol vs K_dil vs KE_total
    ax = axes[0]
    ax.semilogy(t, K_sol, lw=1.6, label=r"$K_{\rm sol}$ (k-eps territory)")
    ax.semilogy(t, np.maximum(K_dil, 1e-30), lw=1.6,
                label=r"$K_{\rm dil}$ (acoustic/host code)")
    ax.semilogy(t, KE, "k-", lw=1.0, alpha=0.5, label=r"$K_{\rm tot}$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("kinetic energy")
    ax.set_title(r"$K_{\rm sol}$ vs $K_{\rm dil}$: classical k-eps assumes $K\approx K_{\rm sol}$")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.5)

    # Panel 2: dK_sol/dt vs -eps_sol (the k-eps balance)
    ax = axes[1]
    m = t > 0.15
    ax.plot(t[m], -dKdt[m], lw=1.6,
            label=r"$-dK_{\rm sol}/dt$ (DNS, smoothed)")
    ax.plot(t[m], eps_sol_s[m], lw=1.6,
            label=r"$\varepsilon_{\rm sol}$ (DNS)")
    ax.plot(t[m], eps_tot_s[m], lw=1.2, ls=":", color="tab:red",
            label=r"$\varepsilon_{\rm tot}$ (DNS)")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\varepsilon$")
    ax.set_title(r"k-eps balance: $-dK_{\rm sol}/dt$ vs $\varepsilon_{\rm sol}$?")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, ls=":", alpha=0.5)

    # Panel 3: power-law fit for C_eps2
    ax = axes[2]
    # Fit log K_sol vs log(t - t_offset) for t in [t_fit_lo, t_fit_hi].
    # Picking t_offset slightly before the fit window so the power law
    # is in the asymptotic regime.
    t_fit_lo, t_fit_hi = 0.30, 1.00
    t_offset = 0.05    # virtual origin (peak time, approximately)
    mfit = (t >= t_fit_lo) & (t <= t_fit_hi) & (K_sol > 0)
    x = np.log(t[mfit] - t_offset)
    y = np.log(K_sol[mfit])
    coeffs = np.polyfit(x, y, 1)
    slope = coeffs[0]      # = -n
    n_fit = -slope
    C_eps2_fit = 1.0 + 1.0 / n_fit if n_fit > 0 else np.nan

    ax.loglog(t - t_offset, K_sol, lw=1.4, label=r"$K_{\rm sol}(t)$ DNS")
    tline = np.array([t_fit_lo, t_fit_hi]) - t_offset
    ax.loglog(tline,
              np.exp(coeffs[1] + slope * np.log(tline)),
              "k--", lw=1.4,
              label=f"fit: slope={slope:.2f}, n={n_fit:.2f}, "
                    + rf"$C_{{\varepsilon 2}}={C_eps2_fit:.2f}$")
    # Reference: standard k-eps slope.
    K_ref = K_sol[mfit][0]
    t_ref = (t[mfit][0] - t_offset)
    n_std = 1.0 / (1.92 - 1.0)   # ~1.087
    ax.loglog(np.array([t_fit_lo, t_fit_hi]) - t_offset,
              K_ref * ((np.array([t_fit_lo, t_fit_hi]) - t_offset)
                       / t_ref) ** (-n_std),
              "r:", lw=1.2,
              label=rf"std k-eps slope ($C_{{\varepsilon 2}}=1.92$, n={n_std:.2f})")
    ax.set_xlabel(r"$t - t_{0}$")
    ax.set_ylabel(r"$K_{\rm sol}$")
    ax.set_title("decay-law fit")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.5)

    fig.suptitle(
        "128$^3$ slip-wall blast: can a k-eps model describe the late stage?",
        y=1.02, fontsize=12)
    out = ROOT / "figs" / "blast_128_keps.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")

    # Print diagnostics.
    print("\nLate-stage k-eps diagnostics (t > 0.3):")
    print(f"  Fit window: t in [{t_fit_lo}, {t_fit_hi}]")
    print(f"  Power-law exponent n  = {n_fit:.3f}")
    print(f"  Implied C_eps2        = {C_eps2_fit:.3f}  (standard = 1.92)")
    for t_check in [0.30, 0.50, 0.75, 1.00]:
        i = int(np.argmin(np.abs(t - t_check)))
        ratio_sol = -dKdt[i] / max(eps_sol_s[i], 1e-30)
        ratio_tot = -dKdt[i] / max(eps_tot[i], 1e-30)
        print(f"  t={t[i]:.3f}: K_sol={K_sol[i]:.4f}, K_dil={K_dil[i]:.4f}, "
              f"-dK_sol/dt={-dKdt[i]:.4f}, eps_sol={eps_sol_s[i]:.4f}, "
              f"-dK_sol/dt / eps_sol = {ratio_sol:.2f}, "
              f"-dK_sol/dt / eps_tot = {ratio_tot:.2f}")


if __name__ == "__main__":
    main()
