#!/usr/bin/env python3
"""Time-averaged solenoidal/dilatational spectra: multifluid vs single-gas blast.
Reads the solver spectra (E_sol, E_dil, E_total per stats step; summed shells)
and compares the developed-turbulence-window averages. Auto-detects 64 vs 128
run dirs under solver/.  Usage: compare_mf_sg_spectra.py [N] [tlo] [thi]
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
N = sys.argv[1] if len(sys.argv) > 1 else "64"
TLO = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
THI = float(sys.argv[3]) if len(sys.argv) > 3 else 0.30
CASES = [("MF (two-gamma)", f"out_blast_{N}_cj_t5/blast_{N}_cj_t5_spectra.h5", "tab:red"),
         ("SG (single gas)", f"out_blast_{N}_sg_t5/blast_{N}_sg_t5_spectra.h5", "tab:blue")]


def avg_spec(path, tlo, thi):
    with h5py.File(path, "r") as f:
        Es, Ed, Et, kk = [], [], [], None
        for key in f.keys():
            g = f[key]; t = float(g["time"][0])
            if tlo <= t <= thi:
                kk = np.asarray(g["k"]); Es.append(np.asarray(g["E_sol"]))
                Ed.append(np.asarray(g["E_dil"])); Et.append(np.asarray(g["E_total"]))
    return kk, np.mean(Es, 0), np.mean(Ed, 0), np.mean(Et, 0), len(Es)


def sl(k, E, klo, khi):
    m = (k >= klo) & (k <= khi) & (E > 0)
    return np.polyfit(np.log(k[m]), np.log(E[m]), 1)[0] if m.sum() > 2 else float("nan")


fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
for label, rel, col in CASES:
    p = run_dir(rel)
    if not p.exists():
        print(f"missing {rel}", file=sys.stderr); continue
    k, Es, Ed, Et, n = avg_spec(p, TLO, THI)
    ssl = sl(k, Es, 4, 12)
    print(f"{label}: {n} spec; K_sol/K_dil={Es.sum()/Ed.sum():.4f}; "
          f"E_sol slope(k=4-12)={ssl:.2f}")
    m = k > 0
    ax[0].loglog(k[m], Es[m], color=col, lw=2, label=f"{label}  $E_{{sol}}$ (slope {ssl:.2f})")
    ax[0].loglog(k[m], Ed[m], color=col, lw=1.2, ls="--", alpha=0.6, label=f"{label}  $E_{{dil}}$")
    # compensated solenoidal
    ax[1].semilogx(k[m], Es[m] * k[m]**(5.0/3.0), color=col, lw=2, label=f"{label}")

kk = np.array([4, 16])
ax[0].loglog(kk, 0.5 * Es.max() * (kk/4.0)**(-5.0/3.0), "k:", lw=1.5, label=r"$k^{-5/3}$")
ax[0].set_xlabel("k"); ax[0].set_ylabel("E(k)  (shell-summed)")
ax[0].set_title(f"Solenoidal & dilatational spectra ({N}$^3$, t$\\in$[{TLO},{THI}])")
ax[0].legend(frameon=False, fontsize=8); ax[0].grid(True, which="both", ls=":", alpha=0.3)
ax[1].set_xlabel("k"); ax[1].set_ylabel(r"$E_{sol}(k)\,k^{5/3}$")
ax[1].set_title("Compensated solenoidal spectrum (flat = Kolmogorov)")
ax[1].legend(frameon=False, fontsize=9); ax[1].grid(True, which="both", ls=":", alpha=0.3)
fig.tight_layout()
out = ROOT / "figs" / f"compare_mf_sg_spectra_{N}.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
