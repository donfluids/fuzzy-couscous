#!/usr/bin/env python3
"""Do the dilatational (compressive) motions feed the solenoidal (vortical) field?

Two complementary, snapshot-based diagnostics, time-resolved:

1) BAROCLINIC enstrophy production  B = < omega . (grad rho x grad p) / rho^2 >.
   This is the canonical compressible -> vortical source: misaligned density and
   pressure gradients (shocks, contact fronts) torque up vorticity, i.e. create
   solenoidal energy out of the thermodynamic/compressible field. B>0 => the
   compressible field is generating vorticity.

2) Advective INTERMODE KE transfer between Helmholtz components. Split the
   velocity u = u_s (solenoidal, div-free) + u_d (dilatational, curl-free) by FFT
   projection. The rate at which advection feeds the solenoidal KE from the
   dilatational velocity is
        Sigma_{d->s} = - < u_s . (u . grad) u_d >,
   and the reverse Sigma_{s->d} = - < u_d . (u . grad) u_s >. Sigma_{d->s}>0 =>
   net KE flux from dilatational into solenoidal motions.

Also tracks K_sol = <1/2 |u_s|^2>, K_dil = <1/2 |u_d|^2>.

Method: spectral (FFT) derivatives and Helmholtz projection on the periodic
extension of the box. CAVEAT: the chamber has slip walls (not periodic), so the
FFT introduces some boundary error -- same approximation the solver uses for its
E_sol/E_dil. Treat magnitudes as indicative; signs/time-trends are robust.

Outputs: figs/sol_dil_transfer.png and sol_dil_transfer.npz.
"""
import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
N = 128
L = 1.0
DX = L / N

# spectral wavenumbers (rad/length) for the unit box
_k1 = 2.0 * np.pi * np.fft.fftfreq(N, d=DX)
KX, KY, KZ = np.meshgrid(_k1, _k1, _k1, indexing="ij")
K2 = KX**2 + KY**2 + KZ**2
K2[0, 0, 0] = 1.0


def ddx(f, K):              # spectral derivative along the axis with wavenumber K
    return np.fft.ifftn(1j * K * np.fft.fftn(f)).real


def helmholtz(u, v, w):
    """Return (u_s, u_d) lists; mean removed. u_d = curl-free, u_s = div-free."""
    U = [u - u.mean(), v - v.mean(), w - w.mean()]
    Uh = [np.fft.fftn(c) for c in U]
    proj = (KX * Uh[0] + KY * Uh[1] + KZ * Uh[2]) / K2     # (k.u_hat)/k^2
    ud = [np.fft.ifftn(KX * proj).real,
          np.fft.ifftn(KY * proj).real,
          np.fft.ifftn(KZ * proj).real]
    us = [U[i] - ud[i] for i in range(3)]
    return us, ud


def adv_dot(u, v, w, target, probe):
    """< probe . (u.grad) target >  for 3-vectors target, probe (lists)."""
    s = 0.0
    for i in range(3):
        adv = (u * ddx(target[i], KX) + v * ddx(target[i], KY)
               + w * ddx(target[i], KZ))
        s += np.mean(probe[i] * adv)
    return float(s)


def load(p):
    with h5py.File(p, "r") as f:
        rho = np.asarray(f["density"], np.float64)
        u = np.asarray(f["velocity_x"], np.float64)
        v = np.asarray(f["velocity_y"], np.float64)
        w = np.asarray(f["velocity_z"], np.float64)
        pr = np.asarray(f["pressure"], np.float64)
        t = float(f["time"][0])
    return t, rho, u, v, w, pr


def snapshots(run_dir, run_name):
    pat = re.compile(rf"{re.escape(run_name)}_(\d+)\.h5$")
    out = []
    for p in sorted(run_dir.glob(f"{run_name}_*.h5")):
        if p.name.endswith(".ckpt.h5") or p.name.endswith("_spectra.h5"):
            continue
        if pat.search(p.name):
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="out_blast_128_budget_seed1")
    ap.add_argument("--name", default="blast_128_budget_seed1")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth snapshot")
    args = ap.parse_args()

    snaps = snapshots(ROOT / args.run, args.name)[:: args.stride]
    if not snaps:
        print("no snapshots", file=sys.stderr); return 1
    print(f"{len(snaps)} snapshots from {args.run}")

    t, Ksol, Kdil, Baro, Sds, Ssd = [], [], [], [], [], []
    for p in snaps:
        tt, rho, u, v, w, pr = load(p)
        us, ud = helmholtz(u, v, w)
        ksol = 0.5 * np.mean(us[0]**2 + us[1]**2 + us[2]**2)
        kdil = 0.5 * np.mean(ud[0]**2 + ud[1]**2 + ud[2]**2)

        # vorticity (full velocity)
        ox = ddx(w, KY) - ddx(v, KZ)
        oy = ddx(u, KZ) - ddx(w, KX)
        oz = ddx(v, KX) - ddx(u, KY)
        # baroclinic vector (grad rho x grad p)/rho^2
        rx, ry, rz = ddx(rho, KX), ddx(rho, KY), ddx(rho, KZ)
        px, py, pz = ddx(pr, KX), ddx(pr, KY), ddx(pr, KZ)
        bx = (ry * pz - rz * py) / rho**2
        by = (rz * px - rx * pz) / rho**2
        bz = (rx * py - ry * px) / rho**2
        baro = float(np.mean(ox * bx + oy * by + oz * bz))

        # advective intermode transfer (advecting velocity = full u)
        sds = -adv_dot(u, v, w, target=ud, probe=us)   # d -> s
        ssd = -adv_dot(u, v, w, target=us, probe=ud)   # s -> d

        t.append(tt); Ksol.append(ksol); Kdil.append(kdil)
        Baro.append(baro); Sds.append(sds); Ssd.append(ssd)
        print(f"  t={tt:.3f}  K_sol={ksol:.3e} K_dil={kdil:.3e}  "
              f"baroclinic={baro:+.3e}  Sigma_d->s={sds:+.3e}  Sigma_s->d={ssd:+.3e}")

    t = np.array(t); Ksol = np.array(Ksol); Kdil = np.array(Kdil)
    Baro = np.array(Baro); Sds = np.array(Sds); Ssd = np.array(Ssd)
    o = np.argsort(t)
    t, Ksol, Kdil, Baro, Sds, Ssd = (a[o] for a in (t, Ksol, Kdil, Baro, Sds, Ssd))

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    a = ax[0]
    a.plot(t, Ksol, lw=2, color="tab:blue", label=r"$K_{sol}$")
    a.plot(t, Kdil, lw=2, color="tab:red", label=r"$K_{dil}$")
    a.set_yscale("log"); a.set_xlabel("t"); a.set_ylabel("KE")
    a.set_title("Helmholtz KE"); a.legend(frameon=False); a.grid(True, which="both", ls=":", alpha=0.4)

    a = ax[1]
    a.plot(t, Baro, lw=2, color="tab:purple")
    a.axhline(0, color="k", lw=0.7)
    a.set_xlabel("t"); a.set_ylabel(r"$\langle\omega\cdot(\nabla\rho\times\nabla p)/\rho^2\rangle$")
    a.set_title("baroclinic enstrophy production\n(>0: compressible field makes vorticity)")
    a.grid(True, ls=":", alpha=0.4)

    a = ax[2]
    a.plot(t, Sds, lw=2, color="tab:green", label=r"$\Sigma_{d\to s}$ (dil$\to$sol)")
    a.plot(t, Ssd, lw=1.6, color="tab:orange", label=r"$\Sigma_{s\to d}$ (sol$\to$dil)")
    a.axhline(0, color="k", lw=0.7)
    a.set_xlabel("t"); a.set_ylabel("advective intermode KE transfer")
    a.set_title("nonlinear transfer between modes\n(>0: into that mode)")
    a.legend(frameon=False); a.grid(True, ls=":", alpha=0.4)

    fig.suptitle(f"Dilatational <-> solenoidal energy exchange ({args.run})", y=1.02)
    fig.tight_layout()
    out = ROOT / "figs" / "sol_dil_transfer.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out}")
    np.savez(DATA / "sol_dil_transfer.npz", t=t, K_sol=Ksol, K_dil=Kdil,
             baroclinic=Baro, Sigma_d_to_s=Sds, Sigma_s_to_d=Ssd)
    print(f"wrote {DATA/'sol_dil_transfer.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
