#!/usr/bin/env python3
"""Paper figure: KE decay in the confined blast, decomposed.

Ensemble (5 budget seeds) total KE(t), dilatational K_dil(t), solenoidal K_sol(t),
log-log with power-law fits K ~ (t-t0)^(-n) over the post-shock window. Shows that
the power-law KE decay is carried by the ACOUSTIC (dilatational) field, while the
vortical (solenoidal) energy is weak and roughly sustained (does not freely decay).
figs/ke_decay_decomposed.png.
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
SEEDS = [1, 2, 3, 4, 5]
WIN = (0.15, 0.5)


def stats(s):
    t, KE = [], []
    p = run_dir(f"out_blast_128_budget_seed{s}") / f"blast_128_budget_seed{s}_stats.csv"
    for r in csv.DictReader(open(p)):
        t.append(float(r["time"])); KE.append(float(r["KE"]))
    return np.array(t), np.array(KE)


def spec(s):
    f = h5py.File(run_dir(f"out_blast_128_budget_seed{s}") / f"blast_128_budget_seed{s}_spectra.h5", "r")
    t, ks, kd = [], [], []
    for key in f.keys():
        g = f[key]; t.append(float(g["time"][0]))
        ks.append(np.asarray(g["E_sol"]).sum()); kd.append(np.asarray(g["E_dil"]).sum())
    o = np.argsort(t)
    return np.array(t)[o], np.array(ks)[o], np.array(kd)[o]


def fit_pl(tt, y, win):
    m = (tt >= win[0]) & (tt <= win[1]) & (y > 0)
    x, yy = tt[m], y[m]
    best = None
    for t0 in np.linspace(-0.05, min(x) - 1e-3, 80):
        lx, ly = np.log(x - t0), np.log(yy)
        A = np.polyfit(lx, ly, 1); res = ly - np.polyval(A, lx)
        r2 = 1 - np.sum(res**2) / np.sum((ly - ly.mean())**2)
        if best is None or r2 > best[0]:
            best = (r2, -A[0], t0, np.exp(A[1]))
    return best  # r2, n, t0, A


def main():
    tg = stats(1)[0]
    KE = np.mean([np.interp(tg, *stats(s)) for s in SEEDS], 0)
    ts = spec(1)[0]
    Ks = np.mean([np.interp(ts, spec(s)[0], spec(s)[1]) for s in SEEDS], 0)
    Kd = np.mean([np.interp(ts, spec(s)[0], spec(s)[2]) for s in SEEDS], 0)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    a = ax[0]
    series = [("total KE", tg, KE, "k"), ("$K_{dil}$ (acoustic)", ts, Kd, "tab:red"),
              ("$K_{sol}$ (vortical)", ts, Ks, "tab:blue")]
    for lab, tt, y, col in series:
        m = (tt > 0.02) & (y > 0)
        a.loglog(tt[m], y[m], lw=2, color=col, label=lab)
        r2, n, t0, A = fit_pl(tt, y, WIN)
        xf = np.linspace(WIN[0], WIN[1], 50)
        a.loglog(xf, A * (xf - t0)**(-n), ls="--", lw=1.4, color=col,
                 label=f"   fit n={n:.2f} (R²={r2:.2f})")
    a.axvspan(*WIN, color="gray", alpha=0.1)
    a.set_xlabel("t"); a.set_ylabel("energy")
    a.set_title("KE decay decomposed (ensemble of 5)\npower law carried by the acoustic field")
    a.legend(frameon=False, fontsize=8); a.grid(True, which="both", ls=":", alpha=0.4)

    a = ax[1]
    a.semilogx(ts[ts > 0.02], (Kd / np.maximum(Ks, 1e-30))[ts > 0.02], lw=2, color="tab:purple")
    a.set_xlabel("t"); a.set_ylabel(r"$K_{dil}/K_{sol}$")
    a.set_title("acoustic dominance vs time\n(K_sol weak & sustained; K_dil decays)")
    a.grid(True, which="both", ls=":", alpha=0.4)
    a.axhline(1.0, color="k", lw=0.7, alpha=0.5)

    fig.suptitle("Confined-blast KE: power-law decay is the acoustic field, "
                 "not vortical turbulence", y=1.02)
    fig.tight_layout()
    out = ROOT / "figs" / "ke_decay_decomposed.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    for lab, tt, y, _ in series:
        r2, n, t0, A = fit_pl(tt, y, WIN)
        print(f"  {lab}: n={n:.2f} t0={t0:+.3f} R2={r2:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
