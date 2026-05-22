#!/usr/bin/env python3
"""Solenoidal content: real TNT (JWL, Atwood~0.999) vs idealized two-gamma
(Atwood~0.82), at IDENTICAL 128^3 closed-chamber numerics.

Reads the two stats CSVs and compares the BC-robust physical-space measures
enstrophy <|omega|^2> (solenoidal) and dilatation <(div u)^2>, the solenoidal
fraction omega2/(omega2+div2), and (where logged) the velocity-Helmholtz
K_sol/K_dil. The question: does the higher Atwood of a real explosive generate
more -- and more sustained -- vortical turbulence, or is the blast still
shock/acoustic (dilatation) dominated?

Usage: python scripts/tnt_vs_tg_solenoidal.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "solver" / "runs"
FIGS = ROOT / "figs" / "tnt"

CASES = [
    ("TNT (JWL, At~0.999)", RUNS / "out_tnt_chamber_128" / "tnt_chamber_128_stats.csv", "C3"),
    ("two-gamma (At~0.82)", RUNS / "out_tg_chamber_128" / "tg_chamber_128_stats.csv", "C0"),
    ("single-fluid (no contact)", RUNS / "out_sf_chamber_128" / "sf_chamber_128_stats.csv", "0.4"),
]


def load(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    return d


def main():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for label, path, col in CASES:
        if not path.exists():
            print(f"missing {path}")
            continue
        d = load(path)
        t, om, dv = d["time"], d["omega2"], d["div2"]
        frac = om / (om + dv + 1e-30)
        ax[0].plot(t, om, col, lw=2, label=label)
        ax[1].plot(t, dv, col, lw=2, label=label)
        ax[2].plot(t, frac, col, lw=2, label=label)
        ip = int(np.argmax(om))
        # K_sol/K_dil where logged (>0 only on spectra steps; serial only)
        khint = ""
        if "K_sol" in d.dtype.names:
            ks, kd = d["K_sol"], d["K_dil"]
            m = (ks > 0) & (kd > 0)
            if m.any():
                khint = f"; K_sol/K_dil final={ks[m][-1] / kd[m][-1]:.3f}"
        print(f"{label}: enstrophy peak={om[ip]:.3e} @ t={t[ip]:.3f} "
              f"(end t={t[-1]:.3f}, om={om[-1]:.3e}); "
              f"sol.frac end={frac[-1]:.3f}, max={frac.max():.3f}{khint}")

    ax[0].set_title("Enstrophy  <|ω|²>  (solenoidal)")
    ax[0].set_xlabel("time"); ax[0].set_ylabel("<|ω|²>")
    ax[1].set_title("Dilatation  <(∇·u)²>")
    ax[1].set_xlabel("time"); ax[1].set_ylabel("<(∇·u)²>")
    ax[2].set_title("Solenoidal fraction  ω²/(ω²+div²)")
    ax[2].set_xlabel("time"); ax[2].set_ylabel("fraction")
    for a in ax:
        a.legend(fontsize=9); a.grid(True, alpha=0.3)

    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "tnt_vs_twogamma_solenoidal.png"
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
