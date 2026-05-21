#!/usr/bin/env python3
"""Compare the blast solenoidal-cascade diagnostics at 128^3 vs 256^3.

Reads the resolution-tagged .npz produced by energy_flux.py (--case 128/256) and
sol_energy_budget.py (--case 128/256), plus K_sol/K_dil(t) straight from the
solver spectra. Produces figs/compare_128_256.png.

Key question: is the 128^3 solenoidal Kolmogorov cascade resolution-converged?
Re_L is physical (same IC), so 256^3 has the SAME turbulence but 2x the resolved
range -> the -5/3 range should widen and the flux=dissipation closure should
persist if the cascade is real.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SPEC = {"128": ROOT / "out_blast_128_budget_seed1" / "blast_128_budget_seed1_spectra.h5",
        "256": ROOT / "out_blast_256_match128" / "blast_256_match128_spectra.h5"}
COL = {"128": "tab:blue", "256": "tab:red"}


def ksol_kdil(spec):
    t, r = [], []
    with h5py.File(spec, "r") as f:
        for key in f.keys():
            g = f[key]
            Es = np.asarray(g["E_sol"]).sum(); Ed = np.asarray(g["E_dil"]).sum()
            if Ed > 0:
                t.append(float(g["time"][0])); r.append(Ed / Es)
    o = np.argsort(t)
    return np.array(t)[o], np.array(r)[o]


def main():
    ef = {}; sb = {}
    for c in ("128", "256"):
        fe = ROOT / f"energy_flux_{c}.npz"; fs = ROOT / f"sol_energy_budget_{c}.npz"
        if fe.exists(): ef[c] = np.load(fe, allow_pickle=True)
        if fs.exists(): sb[c] = np.load(fs, allow_pickle=True)
    if not ef or not sb:
        print("missing npz; run energy_flux.py and sol_energy_budget.py for both cases first", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    # (a) K_dil/K_sol(t)
    a = ax[0, 0]
    for c in ("128", "256"):
        if SPEC[c].exists():
            t, r = ksol_kdil(SPEC[c]); a.plot(t, r, lw=2, color=COL[c], label=f"{c}$^3$")
    a.set_xlabel("t"); a.set_ylabel(r"$K_{dil}/K_{sol}$"); a.set_title("acoustic vs vortical energy")
    a.legend(frameon=False); a.grid(True, ls=":", alpha=0.4); a.set_xlim(0, 0.1)

    # (b) compensated E_sol(k) at t~0.1
    a = ax[0, 1]
    for c in ("128", "256"):
        if c in sb:
            k = sb[c]["k"]; E = sb[c]["E_sol"]; kk = np.arange(len(E))
            m = E > 0
            a.loglog(kk[m][1:], (E[m]*kk[m]**(5/3))[1:], lw=2, color=COL[c],
                     label=f"{c}$^3$ (t~{float(sb[c]['t']):.2f})")
    a.set_xlabel("k (shell)"); a.set_ylabel(r"$E_{sol}(k)\,k^{5/3}$")
    a.set_title(r"compensated $E_{sol}$ (flat = $k^{-5/3}$; wider at 256 if real)")
    a.legend(frameon=False); a.grid(True, which="both", ls=":", alpha=0.4)

    # (c) cascade->dissipation closure
    a = ax[1, 0]
    for c in ("128", "256"):
        if c in sb:
            Pi = sb[c]["Pi"]; Dab = sb[c]["D_above"]; kk = np.arange(len(Pi))
            ratio = float(sb[c]["plateau"]) / float(sb[c]["Dtot"])
            a.plot(kk[1:], Pi[1:], lw=2.2, color=COL[c], label=f"{c}$^3$ $\\Pi_{{sol}}$ (closure {ratio:.2f})")
            a.plot(kk[1:], Dab[1:], lw=1.4, color=COL[c], ls="--", alpha=0.7)
    a.axhline(0, color="k", lw=0.7); a.set_xscale("log")
    a.set_xlabel("k (shell)"); a.set_ylabel(r"$\Pi_{sol}$ (solid), $\sum_{k'>k}D$ (dashed)")
    a.set_title("cascade flux vs downstream dissipation")
    a.legend(frameon=False, fontsize=8); a.grid(True, which="both", ls=":", alpha=0.4)

    # (d) E_sol slope vs time + onset
    a = ax[1, 1]
    for c in ("128", "256"):
        if c in ef:
            a.plot(ef[c]["slope_t"], ef[c]["slope"], lw=2, color=COL[c], label=f"{c}$^3$")
            on = float(ef[c]["onset"])
            if on > 0: a.axvline(on, color=COL[c], ls=":", lw=1.2)
    a.axhline(-5/3, color="green", ls="--", lw=1.4, label="-5/3")
    a.axhspan(-2.1, -1.3, color="green", alpha=0.1)
    a.set_ylim(-4, 1); a.set_xlim(0, 0.1); a.set_xlabel("t")
    a.set_ylabel(r"$E_{sol}$ slope (k=8-40)"); a.set_title("cascade onset (slope vs time)")
    a.legend(frameon=False, fontsize=8); a.grid(True, ls=":", alpha=0.4)

    fig.suptitle("Blast solenoidal cascade: 128$^3$ vs 256$^3$ (identical physical IC)", y=1.0)
    fig.tight_layout()
    out = ROOT / "figs" / "compare_128_256.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")
    for c in ("128", "256"):
        if c in sb:
            print(f"  {c}^3: cascade plateau={float(sb[c]['plateau']):.3e}  "
                  f"total D_sol={float(sb[c]['Dtot']):.3e}  closure={float(sb[c]['plateau'])/float(sb[c]['Dtot']):.2f}  "
                  f"onset={float(ef[c]['onset']) if c in ef else float('nan'):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
