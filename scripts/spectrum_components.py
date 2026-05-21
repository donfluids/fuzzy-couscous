#!/usr/bin/env python3
"""Is the recovered 'Kolmogorov' spectrum vortical, acoustic, or neither?

Decomposes the time-averaged blast spectrum into solenoidal (vortical) and
dilatational (acoustic) parts and reads off the LOCAL slope d(log E)/d(log k)
for each, so we can see (a) which component carries any k^-5/3 range, (b) over
how many octaves, and (c) whether it is a genuine plateau or a transitional
blend.

Single seed, narrow time-average window (default the dense early run seed 1 over
t in [0.01,0.02]) to avoid smearing an evolving spectrum.

Context: the resolved vortical turbulence here is weak (Re_L ~ 20-90, solenoidal
energy fraction ~ 1-10%). Classical k^-5/3 needs a STRONG high-Re vortical
cascade, so weakness argues against a true inertial range, not for one.
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def spectra_path(tag, seed):
    return ROOT / f"out_blast_128_{tag}_seed{seed}" / f"blast_128_{tag}_seed{seed}_spectra.h5"


def tavg(path, win):
    k0 = None
    Es = Ed = Et = None
    n = 0
    tlo, thi = 1e9, -1e9
    with h5py.File(path, "r") as f:
        for key in f.keys():
            tt = float(f[key]["time"][0])
            if not (win[0] <= tt <= win[1]):
                continue
            g = f[key]
            k = np.asarray(g["k"], np.float64)
            if k0 is None:
                k0 = k; Es = np.zeros_like(k); Ed = np.zeros_like(k); Et = np.zeros_like(k)
            Es += np.asarray(g["E_sol"]); Ed += np.asarray(g["E_dil"]); Et += np.asarray(g["E_total"])
            n += 1
            tlo, thi = min(tlo, tt), max(thi, tt)
    if n == 0:
        raise SystemExit(f"no spectra in window {win} of {path}")
    return k0, Es / n, Ed / n, Et / n, n, (tlo, thi)


def local_slope(k, E):
    m = (k > 0) & (E > 0)
    lk = np.log(k[m]); lE = np.log(E[m])
    return k[m], np.gradient(lE, lk)


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
    print(f"seed {args.seed} ({args.tag}), window {win} -> {n} spectra "
          f"(actual t in [{tlo:.4f},{thi:.4f}])")

    # energy split
    m = k > 0
    Ksol = np.trapz(Es[m], k[m]); Kdil = np.trapz(Ed[m], k[m]); Ktot = np.trapz(Et[m], k[m])
    print(f"  K_sol/K_tot = {Ksol/Ktot:.4f}   K_dil/K_sol = {Kdil/Ksol:.1f}")
    ipk = np.argmax(Et[1:]) + 1
    print(f"  E_total peak at k={k[ipk]:.2f} (2*pi={2*np.pi:.2f}) "
          f"-- dilatational fraction there = {Ed[ipk]/Et[ipk]:.3f}")

    print("\n   k      E_total     E_sol      E_dil    E_sol/E_tot")
    for kk in (3, 6, 9, 16, 31, 60, 120, 250):
        i = int(np.argmin(np.abs(k - kk)))
        print(f"  {k[i]:5.1f}  {Et[i]:.3e}  {Es[i]:.3e}  {Ed[i]:.3e}   {Es[i]/max(Et[i],1e-30):.3f}")

    print("\n  local slope near -5/3 (-1.67):")
    for name, E in [("E_total", Et), ("E_sol", Es), ("E_dil", Ed)]:
        kk, sl = local_slope(k, E)
        near = np.abs(sl + 5.0 / 3.0) < 0.25
        kr = kk[near]
        rng = f"k={kr.min():.1f}-{kr.max():.1f} ({kr.size} bins)" if kr.size else "none"
        print(f"   {name}: slope@k6={np.interp(6,kk,sl):+.2f} slope@k16={np.interp(16,kk,sl):+.2f}"
              f"  within 0.25 of -5/3: {rng}")

    # ---------------------------------------------------------------- plots
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    cm = {"E_total": "k", "E_sol": "tab:blue", "E_dil": "tab:red"}

    a = ax[0]
    for name, E in [("E_total", Et), ("E_sol", Es), ("E_dil", Ed)]:
        mm = (k > 0) & (E > 0)
        a.loglog(k[mm], E[mm] * k[mm] ** (5.0 / 3.0), lw=2, color=cm[name], label=name)
    a.set_xlabel("k"); a.set_ylabel(r"$E(k)\,k^{5/3}$")
    a.set_title(r"compensated spectra (flat $\Rightarrow k^{-5/3}$)")
    a.legend(frameon=False); a.grid(True, which="both", ls=":", alpha=0.4)
    a.set_xlim(1.5, k.max())

    a = ax[1]
    for name, E in [("E_total", Et), ("E_sol", Es), ("E_dil", Ed)]:
        kk, sl = local_slope(k, E)
        a.semilogx(kk, sl, lw=2, color=cm[name], label=name)
    a.axhline(-5.0 / 3.0, color="green", ls="--", lw=1.4, label=r"$-5/3$")
    a.set_ylim(-4, 3); a.set_xlim(1.5, k.max())
    a.set_xlabel("k"); a.set_ylabel(r"local slope $d\log E/d\log k$")
    a.set_title("local spectral slope"); a.legend(frameon=False, fontsize=8)
    a.grid(True, which="both", ls=":", alpha=0.4)

    fig.suptitle(f"Blast spectrum decomposition  seed {args.seed}  "
                 f"t-avg [{tlo:.3f},{thi:.3f}]  ({n} spectra)", y=1.0)
    fig.tight_layout()
    out = ROOT / "figs" / f"blast_spectrum_components_seed{args.seed}_{args.t0:.3f}_{args.t1:.3f}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
