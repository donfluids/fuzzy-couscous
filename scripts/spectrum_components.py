#!/usr/bin/env python3
"""Dilatational *energy* vs dilatational *turbulence* in the blast spectrum.

Splits the time-averaged blast spectrum into solenoidal (vortical) and
dilatational (compressible) parts and reports the two along separate axes:

  * **Energy (magnitude):** the integrated K_sol / K_dil and their fractions --
    how much kinetic energy is compressible. A large K_dil alone says nothing
    about turbulence (a single standing compression carries dilatational energy).

  * **Turbulence (spectral shape):** the LOCAL slope d(log E)/d(log k) and the
    width of any contiguous power-law range -- whether a component actually
    cascades, and over how many octaves, vs. being peaked / energy-containing.

The shape/energy helpers live in tools/spectrum_shape.py so this script,
total_spectrum_windows.py, and tools/plot_spectra.py share one implementation.

Single seed, narrow time-average window (default the dense early run seed 1 over
t in [0.01,0.02]) to avoid smearing an evolving spectrum.

Context: the resolved vortical turbulence here is weak (Re_L ~ 20-90, solenoidal
energy fraction ~ 1-10%). Classical k^-5/3 needs a STRONG high-Re vortical
cascade, so weakness argues against a true inertial range, not for one -- which
is exactly why energy (lots of dilatational) must not be read as turbulence
(no contiguous cascade).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

# Shared spectrum-shape utility. tavg / local_slope are re-exported here so that
# `from spectrum_components import tavg, local_slope` (total_spectrum_windows.py)
# keeps working.
sys.path.insert(0, str(ROOT / "tools"))
from spectrum_shape import (  # noqa: E402
    tavg, local_slope, smooth_slope, integrated_energy, peak_k, slope_at,
    powerlaw_range, classify_shape,
)

TARGET = -5.0 / 3.0
TOL = 0.25
SLOPE_FLOOR = 1e-6  # mask the slope plot below this fraction of the peak
SLOPE_SMOOTH = 0.4  # half-width in ln(k) of the slope-plot smoothing fit


def spectra_path(tag, seed):
    return RUNS / f"out_blast_128_{tag}_seed{seed}" / f"blast_128_{tag}_seed{seed}_spectra.h5"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="early", help="run set (early|budget)")
    ap.add_argument("--t0", type=float, default=0.01)
    ap.add_argument("--t1", type=float, default=0.02)
    args = ap.parse_args()
    win = (args.t0, args.t1)

    path = spectra_path(args.tag, args.seed)
    k, Es, Ed, Et, n, (tlo, thi) = tavg(path, win)
    spec = {"E_total": Et, "E_sol": Es, "E_dil": Ed}
    print(f"seed {args.seed} ({args.tag}), window {win} -> {n} spectra "
          f"(actual t in [{tlo:.4f},{thi:.4f}])")

    # ---- energy: magnitude / bookkeeping only ---------------------------
    Ktot = integrated_energy(k, Et)
    Ksol = integrated_energy(k, Es)
    Kdil = integrated_energy(k, Ed)
    print("\n  ENERGY (magnitude): how much kinetic energy is compressible")
    print(f"    K_sol/K_tot = {Ksol/Ktot:.4f}   K_dil/K_tot = {Kdil/Ktot:.4f}"
          f"   K_dil/K_sol = {Kdil/max(Ksol,1e-30):.1f}")
    ipk = np.argmax(Et[1:]) + 1
    print(f"    E_total peak at k={k[ipk]:.2f} (2*pi={2*np.pi:.2f}) "
          f"-- dilatational fraction there = {Ed[ipk]/Et[ipk]:.3f}")

    print("\n   k      E_total     E_sol      E_dil    E_sol/E_tot")
    for kk in (3, 6, 9, 16, 31, 60, 120, 250):
        i = int(np.argmin(np.abs(k - kk)))
        print(f"  {k[i]:5.1f}  {Et[i]:.3e}  {Es[i]:.3e}  {Ed[i]:.3e}   "
              f"{Es[i]/max(Et[i],1e-30):.3f}")

    # ---- turbulence: spectral shape / cascade only ----------------------
    print(f"\n  TURBULENCE (spectral shape): contiguous power-law range near "
          f"{TARGET:+.2f} (tol {TOL})")
    for name in ("E_total", "E_sol", "E_dil"):
        E = spec[name]
        rng = powerlaw_range(k, E, TARGET, TOL)
        rng_s = (f"k={rng['k_lo']:.1f}-{rng['k_hi']:.1f} "
                 f"({rng['octaves']:.1f} oct, {rng['nbins']} bins)"
                 if rng["nbins"] else "none")
        verdict = classify_shape(k, E, TARGET, TOL)
        print(f"   {name}: slope@k6={slope_at(k,E,6):+.2f} "
              f"slope@k16={slope_at(k,E,16):+.2f}  range: {rng_s}  -> {verdict}")

    # ---------------------------------------------------------------- plots
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    cm = {"E_total": "k", "E_sol": "tab:blue", "E_dil": "tab:red"}

    a = ax[0]
    for name in ("E_total", "E_sol", "E_dil"):
        E = spec[name]
        mm = (k > 0) & (E > 0)
        a.loglog(k[mm], E[mm] * k[mm] ** (5.0 / 3.0), lw=2, color=cm[name], label=name)
    a.set_xlabel("k"); a.set_ylabel(r"$E(k)\,k^{5/3}$")
    a.set_title(r"ENERGY: compensated spectra (flat $\Rightarrow k^{-5/3}$)")
    a.legend(frameon=False); a.grid(True, which="both", ls=":", alpha=0.4)
    a.set_xlim(1.5, k.max())

    a = ax[1]
    for name in ("E_total", "E_sol", "E_dil"):
        kk, sl = smooth_slope(k, spec[name], dlnk=SLOPE_SMOOTH, rel_floor=SLOPE_FLOOR)
        a.semilogx(kk, sl, lw=2, color=cm[name], label=name)
    a.axhline(TARGET, color="green", ls="--", lw=1.4, label=r"$-5/3$")
    a.set_ylim(-4, 3); a.set_xlim(1.5, k.max())
    a.set_xlabel("k"); a.set_ylabel(r"local slope $d\log E/d\log k$")
    a.set_title("TURBULENCE: local spectral slope"); a.legend(frameon=False, fontsize=8)
    a.grid(True, which="both", ls=":", alpha=0.4)

    fig.suptitle(f"Blast spectrum: energy vs turbulence  seed {args.seed}  "
                 f"t-avg [{tlo:.3f},{thi:.3f}]  ({n} spectra)", y=1.0)
    fig.tight_layout()
    out = ROOT / "figs" / f"blast_spectrum_components_seed{args.seed}_{args.t0:.3f}_{args.t1:.3f}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
