#!/usr/bin/env python3
"""Does the hybrid earn its keep? Compare a COARSE LES and a COARSE HYBRID
(LES+BHR+f_k) against a FINE LES reference, same physical blast.

Reference : out_blast_128_fineLES  (128^3 pure LES)
Coarse LES: out_blast_64_coarseLES (64^3 pure LES)
Coarse hyb: out_blast_64_hybridV   (64^3 LES + BHR, f_k-blended feedback)

Metric: on the wavenumbers the coarse grid resolves, is the coarse-HYBRID
spectrum closer to the fine LES than the coarse-LES is? Plus KE(t)/tke(t).
Spectra bins share k_fund (L=1) across resolutions, so bin index b == same
physical k; compare bin-by-bin up to the coarse Nyquist.
figs/hybrid_validation.png.
"""
import csv
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUNS = {
    "fine LES (128)":  ("out_blast_128_fineLES",  "blast_128_fineLES",  "k"),
    "coarse LES (64)": ("out_blast_64_coarseLES", "blast_64_coarseLES", "tab:orange"),
    "coarse hybrid (64)": ("out_blast_64_hybridV", "blast_64_hybridV",  "tab:green"),
}
TCMP = 0.10


def load_stats(d, n):
    t, KE, tke = [], [], []
    with open(ROOT / "runs" / d / f"{n}_stats.csv") as f:
        for r in csv.DictReader(f):
            t.append(float(r["time"])); KE.append(float(r["KE"])); tke.append(float(r["tke"]))
    return np.array(t), np.array(KE), np.array(tke)


def load_spec(d, n, tt):
    p = ROOT / "runs" / d / f"{n}_spectra.h5"
    with h5py.File(p, "r") as f:
        key = min(f.keys(), key=lambda k: abs(float(f[k]["time"][0]) - tt))
        g = f[key]
        return (float(g["time"][0]), np.asarray(g["k"]),
                np.asarray(g["E_total"]), np.asarray(g["E_sol"]))


def main():
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))

    # (1) KE(t), tke(t)
    fineKE = None
    for lab, (d, n, col) in RUNS.items():
        try:
            t, KE, tke = load_stats(d, n)
        except FileNotFoundError:
            print(f"missing stats for {lab}", file=sys.stderr); continue
        ax[0].plot(t, KE, lw=2, color=col, label=lab + " KE")
        ax[0].plot(t, tke, lw=1.2, ls=":", color=col)
    ax[0].set_xlabel("t"); ax[0].set_ylabel("KE (solid), tke (dotted)")
    ax[0].set_title("resolved energy vs time"); ax[0].legend(frameon=False, fontsize=8)
    ax[0].grid(True, ls=":", alpha=0.4)

    # (2) spectra at TCMP + (3) error vs fine
    specs = {}
    for lab, (d, n, col) in RUNS.items():
        try:
            specs[lab] = load_spec(d, n, TCMP)
        except (FileNotFoundError, OSError):
            print(f"missing spectra for {lab}", file=sys.stderr)
    if "fine LES (128)" not in specs:
        print("no fine reference yet (still running?)", file=sys.stderr)
    for lab, (col) in [(l, RUNS[l][2]) for l in specs]:
        tt, k, Et, Es = specs[lab]
        m = (k > 0) & (Es > 0)
        ax[1].loglog(k[m], Es[m], lw=2, color=col, label=f"{lab} (t={tt:.2f})")
    ax[1].set_xlabel("k"); ax[1].set_ylabel(r"$E_{sol}(k)$")
    ax[1].set_title(f"solenoidal spectrum at t~{TCMP}")
    ax[1].legend(frameon=False, fontsize=8); ax[1].grid(True, which="both", ls=":", alpha=0.4)

    # error metric: bin-by-bin vs fine on resolved range
    if "fine LES (128)" in specs:
        _, kf, _, Ef = specs["fine LES (128)"]
        for lab in ("coarse LES (64)", "coarse hybrid (64)"):
            if lab not in specs:
                continue
            _, kc, _, Ec = specs[lab]
            nb = min(len(Ef), len(Ec))
            b = np.arange(nb)
            lo, hi = 4, nb - 1            # resolved band up to coarse Nyquist
            band = (b >= lo) & (b <= hi) & (Ef[:nb] > 0)
            # relative L2 error of E_sol vs fine
            err = np.sqrt(np.mean(((Ec[:nb][band] - Ef[:nb][band]) / Ef[:nb][band])**2))
            ax[2].plot(b[band], Ec[:nb][band]/Ef[:nb][band], lw=2,
                       color=RUNS[lab][2], label=f"{lab}: rms rel.err={err:.2f}")
            print(f"  {lab}: E_sol rms relative error vs fine = {err:.3f}")
        ax[2].axhline(1.0, color="k", lw=0.8)
        ax[2].set_xlabel("k bin"); ax[2].set_ylabel(r"$E_{sol}/E_{sol}^{fine}$")
        ax[2].set_title("ratio to fine (1=perfect)")
        ax[2].legend(frameon=False, fontsize=8); ax[2].grid(True, ls=":", alpha=0.4)
        ax[2].set_ylim(0, 3)

    fig.suptitle("Hybrid validation: coarse LES vs coarse HYBRID vs fine LES", y=1.02)
    fig.tight_layout()
    out = ROOT / "figs" / "hybrid_validation.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
