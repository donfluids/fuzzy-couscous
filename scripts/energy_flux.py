#!/usr/bin/env python3
"""When does the solenoidal (vortical) field develop a Kolmogorov cascade?
(N-agnostic; --case 128 = budget 5-seed ensemble, --case 256 = match128 run.)

Signatures:
  (1) E_sol(k) ~ k^-5/3 over an inertial range (compensated spectrum flat).
  (2) constant positive (forward) spectral energy FLUX Pi_sol(k) over that range.

  T_s(k) = (1/N^6) sum_{|k| in shell} Re[ u_s_hat* . FFT(-(U.grad)U) ]
  Pi_sol(k) = sum_{k'>k} T_s(k')    (forward = energy to small scales)

E_sol(k) slope-vs-time is read from the solver's own spectra to pinpoint onset.
Uses the VALIDATED periodic-FFT Helmholtz (intermode_transfer).
Outputs figs/energy_flux_<case>.png and energy_flux_<case>.npz.
"""
import argparse
import glob
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from intermode_transfer import ddx, load_xyz, helmholtz, kbin, nbin, kphys  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
DATA = ROOT / "data"
TARGET_TIMES = [0.03, 0.05, 0.08, 0.10]
KFIT = (8, 40)


def case_runs(case):
    """Return (list of (dir,name) for snapshots, spectra_path_for_slope)."""
    if case == "128":
        runs = [(f"out_blast_128_budget_seed{s}", f"blast_128_budget_seed{s}")
                for s in (1, 2, 3, 4, 5)]
        spec = RUNS / "out_blast_128_budget_seed1" / "blast_128_budget_seed1_spectra.h5"
    elif case == "256":
        runs = [("out_blast_256_match128", "blast_256_match128")]
        spec = RUNS / "out_blast_256_match128" / "blast_256_match128_spectra.h5"
    else:
        raise SystemExit(f"unknown case {case}")
    return runs, spec


def snaps(d, name):
    out = []
    for p in sorted(glob.glob(str(RUNS / d / f"{name}_*.h5"))):
        if p.endswith(".ckpt.h5") or p.endswith("_spectra.h5"):
            continue
        with h5py.File(p, "r") as f:
            out.append((float(f["time"][0]), p))
    return out


def solenoidal_flux(p):
    t, rho, u, v, w = load_xyz(p)
    N = u.shape[0]
    us, ud = helmholtz(u, v, w)
    U = [us[i] + ud[i] for i in range(3)]
    NL = [-(U[0]*ddx(U[i], 0) + U[1]*ddx(U[i], 1) + U[2]*ddx(U[i], 2))
          for i in range(3)]
    KB = kbin(N); nb = nbin(N)
    Ts = np.zeros(nb); Es = np.zeros(nb)
    for i in range(3):
        ush = np.fft.fftn(us[i]); nlh = np.fft.fftn(NL[i])
        Ts += np.bincount(KB.ravel(), weights=(np.real(np.conj(ush)*nlh)/N**6).ravel(), minlength=nb)[:nb]
        Es += np.bincount(KB.ravel(), weights=(0.5*np.abs(ush)**2/N**6).ravel(), minlength=nb)[:nb]
    Pi = np.array([Ts[k+1:].sum() for k in range(nb)])
    return t, Es, Pi, N


def slope_timeseries(spec):
    t, sl = [], []
    klo, khi = KFIT
    with h5py.File(spec, "r") as f:
        for key in f.keys():
            g = f[key]
            Es = np.asarray(g["E_sol"])
            kk = np.arange(len(Es))
            m = (kk >= klo) & (kk <= khi) & (Es > 0)
            if m.sum() < 4:
                continue
            t.append(float(g["time"][0]))
            sl.append(np.polyfit(np.log(kk[m]), np.log(Es[m]), 1)[0])
    o = np.argsort(t)
    return np.array(t)[o], np.array(sl)[o]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="256", choices=["128", "256"])
    args = ap.parse_args()
    runs, spec = case_runs(args.case)

    results = []
    for tt in TARGET_TIMES:
        Es_acc = Pi_acc = None; tav = 0.0; n = 0; Nout = None
        for d, name in runs:
            sn = snaps(d, name)
            if not sn:
                continue
            t0, p = min(sn, key=lambda tp: abs(tp[0] - tt))
            t, Es, Pi, N = solenoidal_flux(p); Nout = N
            Es_acc = Es if Es_acc is None else Es_acc + Es
            Pi_acc = Pi if Pi_acc is None else Pi_acc + Pi
            tav += t; n += 1
        if not n:
            continue
        Es_acc /= n; Pi_acc /= n; tav /= n
        results.append((tav, Es_acc, Pi_acc))
        kk = np.arange(len(Pi_acc)); band = (kk >= KFIT[0]) & (kk <= KFIT[1])
        print(f"  t={tav:.3f}  Pi_sol[band]={Pi_acc[band].mean():+.3e}  "
              f"flatness={Pi_acc[band].std()/abs(Pi_acc[band].mean()+1e-30):.2f}")

    ts, sl = slope_timeseries(spec)
    onset = next((ts[i] for i in range(len(ts))
                  if -2.1 <= sl[i] <= -1.3 and ts[i] > 0.02), None)
    print(f"\n  case {args.case} (N~{Nout}): E_sol slope reaches ~-5/3 at t ~ {onset}")

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    cm = plt.cm.viridis(np.linspace(0, 0.9, len(results)))
    for c, (t, Es, Pi) in zip(cm, results):
        kk = np.arange(len(Pi))
        ax[0].plot(kk[1:int(0.5*len(kk))], Pi[1:int(0.5*len(kk))], lw=2, color=c, label=f"t={t:.2f}")
        m = Es > 0
        ax[1].loglog(kk[m][1:int(0.5*len(kk))], (Es[m]*kk[m]**(5/3))[1:int(0.5*len(kk))], lw=2, color=c, label=f"t={t:.2f}")
    for a in (ax[0], ax[1]):
        a.axvspan(KFIT[0], KFIT[1], color="gray", alpha=0.12); a.grid(True, which="both", ls=":", alpha=0.4)
    ax[0].axhline(0, color="k", lw=0.7); ax[0].set_xscale("log")
    ax[0].set_xlabel("k"); ax[0].set_ylabel(r"$\Pi_{sol}(k)$"); ax[0].set_title("solenoidal flux")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].set_xlabel("k"); ax[1].set_ylabel(r"$E_{sol}k^{5/3}$"); ax[1].set_title("compensated $E_{sol}$")
    ax[1].legend(frameon=False, fontsize=8)
    ax[2].plot(ts, sl, lw=2, color="tab:blue"); ax[2].axhline(-5/3, color="green", ls="--", lw=1.4, label="-5/3")
    ax[2].axhspan(-2.1, -1.3, color="green", alpha=0.1)
    if onset: ax[2].axvline(onset, color="red", ls=":", lw=1.2, label=f"onset~{onset:.2f}")
    ax[2].set_ylim(-4, 1); ax[2].set_xlabel("t"); ax[2].set_ylabel(f"$E_{{sol}}$ slope (k={KFIT[0]}-{KFIT[1]})")
    ax[2].set_title("slope vs time"); ax[2].legend(frameon=False, fontsize=8); ax[2].grid(True, ls=":", alpha=0.4)
    fig.suptitle(f"Solenoidal cascade, case {args.case} (N~{Nout})", y=1.02)
    fig.tight_layout()
    out = ROOT / "figs" / f"energy_flux_{args.case}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); print(f"wrote {out}")
    np.savez(DATA / f"energy_flux_{args.case}.npz",
             times=[r[0] for r in results], E_sol=[r[1] for r in results],
             Pi_sol=[r[2] for r in results], slope_t=ts, slope=sl,
             onset=onset if onset else -1, N=Nout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
