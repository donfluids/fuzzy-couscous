#!/usr/bin/env python3
"""Solenoidal energy flow: source -> cascade -> dissipation. (N-agnostic.)
--case 128 (budget 5-seed ensemble) | 256 (match128).

Robust, checkable cascade->dissipation balance: in/below the inertial range the
solenoidal energy flux through k equals the dissipation at all smaller scales,
    Pi_sol(k) ~= sum_{k'>k} D_sol(k').
  Pi_sol(k) = sum_{k'>k} T_s(k'),  T_s = (1/N^6)Re[u_s_hat* . FFT(-(U.grad)U)]
  D_sol(k)  = 2 (nu k^2 + nu6 k^6) E_sol(k)
The plateau Pi_sol ~= total D_sol == Kolmogorov flux=epsilon balance. The source
that sets the level is baroclinic injection (baroclinic_source.py).
Outputs figs/sol_energy_budget_<case>.png, sol_energy_budget_<case>.npz.
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
MU = 5.0e-4
WIN = (0.08, 0.10)


def hyper6(case):
    return 2.5e-14 if case == "128" else 3.9e-16


def case_runs(case):
    if case == "128":
        return [(f"out_blast_128_budget_seed{s}", f"blast_128_budget_seed{s}")
                for s in (1, 2, 3, 4, 5)]
    return [("out_blast_256_match128", "blast_256_match128")]


def one(p, nu6):
    t, rho, u, v, w = load_xyz(p)
    N = u.shape[0]
    us, ud = helmholtz(u, v, w)
    U = [us[i] + ud[i] for i in range(3)]
    NL = [-(U[0]*ddx(U[i], 0) + U[1]*ddx(U[i], 1) + U[2]*ddx(U[i], 2)) for i in range(3)]
    KB = kbin(N); nb = nbin(N); kp = kphys(N)
    Ts = np.zeros(nb); Es = np.zeros(nb)
    for i in range(3):
        ush = np.fft.fftn(us[i]); nlh = np.fft.fftn(NL[i])
        Ts += np.bincount(KB.ravel(), weights=(np.real(np.conj(ush)*nlh)/N**6).ravel(), minlength=nb)[:nb]
        Es += np.bincount(KB.ravel(), weights=(0.5*np.abs(ush)**2/N**6).ravel(), minlength=nb)[:nb]
    nu = MU / rho.mean()
    Ds = 2.0 * (nu * kp**2 + nu6 * kp**6) * Es
    return t, Ts, Es, Ds, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="256", choices=["128", "256"])
    args = ap.parse_args()
    nu6 = hyper6(args.case)

    Ts = Es = Ds = None; tav = 0.0; n = 0; N = None
    for d, name in case_runs(args.case):
        for p in sorted(glob.glob(str(ROOT / d / f"{name}_*.h5"))):
            if p.endswith(".ckpt.h5") or p.endswith("_spectra.h5"):
                continue
            with h5py.File(p, "r") as f:
                t = float(f["time"][0])
            if not (WIN[0] <= t <= WIN[1]):
                continue
            _, T, E, D, N = one(p, nu6)
            Ts = T if Ts is None else Ts + T
            Es = E if Es is None else Es + E
            Ds = D if Ds is None else Ds + D
            tav += t; n += 1
    if not n:
        print("no snapshots in window", file=sys.stderr); return 1
    Ts /= n; Es /= n; Ds /= n; tav /= n
    nb = len(Ts)
    Pi = np.array([Ts[k+1:].sum() for k in range(nb)])
    Dabove = np.array([Ds[k+1:].sum() for k in range(nb)])
    Dtot = Ds.sum()
    plateau = Pi[6:26].mean()
    print(f"case {args.case} N={N}: {n} snapshots, t~{tav:.3f}")
    print(f"  total D_sol={Dtot:.3e}  cascade plateau(k6-25)={plateau:.3e}  "
          f"ratio={plateau/Dtot:.2f}")

    kk = np.arange(nb); hi = int(0.55 * nb)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    ax[0].loglog(kk[1:hi], Es[1:hi], lw=2, color="tab:blue", label=r"$E_{sol}(k)$")
    ax[0].loglog(kk[1:hi], Ds[1:hi], lw=2, color="tab:red", label=r"$D_{sol}(k)$")
    ax[0].set_xlabel("k"); ax[0].set_ylabel("spectral density")
    ax[0].set_title("energy@large, dissipation@small"); ax[0].legend(frameon=False)
    ax[0].grid(True, which="both", ls=":", alpha=0.4)
    ax[1].plot(kk[1:hi], Pi[1:hi], lw=2.4, color="tab:blue", label=r"$\Pi_{sol}(k)$")
    ax[1].plot(kk[1:hi], Dabove[1:hi], lw=2, color="tab:red", ls="--", label=r"$\sum_{k'>k}D_{sol}$")
    ax[1].axhline(Dtot, color="gray", ls=":", lw=1, label="total dissipation")
    ax[1].axhline(0, color="k", lw=0.7); ax[1].set_xscale("log")
    ax[1].set_xlabel("k"); ax[1].set_ylabel("flux / cumulative diss.")
    ax[1].set_title(rf"cascade feeds dissipation (ratio {plateau/Dtot:.2f})")
    ax[1].legend(frameon=False, fontsize=8); ax[1].grid(True, which="both", ls=":", alpha=0.4)
    fig.suptitle(f"Solenoidal cascade->dissipation, case {args.case} (N={N}, t~{tav:.2f})", y=1.02)
    fig.tight_layout()
    out = ROOT / "figs" / f"sol_energy_budget_{args.case}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); print(f"wrote {out}")
    np.savez(ROOT / f"sol_energy_budget_{args.case}.npz", k=kphys(N), T_s=Ts, Pi=Pi,
             E_sol=Es, D_sol=Ds, D_above=Dabove, plateau=plateau, Dtot=Dtot, t=tav, N=N)
    return 0


if __name__ == "__main__":
    sys.exit(main())
