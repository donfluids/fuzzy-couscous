#!/usr/bin/env python3
"""Scale-resolved nonlinear energy transfer between dilatational and solenoidal
modes:  T_{d->s}(k).   (N-agnostic: works at 128^3, 256^3, ...)

Helmholtz-split the velocity u = u_s + u_d (periodic FFT -- VALIDATED to
reproduce the solver's E_sol/E_dil exactly once the HDF5 [z,y,x] storage order
is corrected to [x,y,z]). The advective inter-mode transfer INTO the solenoidal
field from the dilatational velocity is

    Sigma_{d->s} = < u_s . A >,   A = -(u.grad) u_d        (global)
    T_{d->s}(k)  = (1/N^6) sum_{|kappa| in shell k} Re[ u_s_hat* . A_hat ]   (spectral)

so that sum_k T_{d->s}(k) = Sigma_{d->s}. T>0 at scale k means dilatational
motions feed solenoidal KE there. This is the nonlinear scattering channel; the
baroclinic torque (scripts/baroclinic_source.py) is the complementary
vorticity-generation channel.

CAVEAT: periodic-FFT basis (what the serial solver's diagnostics use); the slip
walls make this an approximation near the boundary.

This module also exposes the shared, N-agnostic primitives used by
energy_flux.py and sol_energy_budget.py:
  load_xyz(p), helmholtz(u,v,w), ddx(f,axis), grids(N) -> dict, and
  kbin(N)/nbin(N)/kphys(N).
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

import sys
from pathlib import Path
_PP = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").is_file())
for _d in (_PP, _PP / "tools", _PP / "scripts"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
from paths import REPO_ROOT as ROOT, RUNS, DATA, FIGS, run_dir  # noqa: E402

# ---- N-agnostic FFT grids (cached per resolution) -------------------------
_CACHE = {}


def grids(N):
    if N not in _CACHE:
        k1 = 2.0 * np.pi * np.fft.fftfreq(N, d=1.0 / N)
        KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
        K2 = KX**2 + KY**2 + KZ**2
        K2[0, 0, 0] = 1.0
        KBIN = np.round(np.sqrt(KX**2 + KY**2 + KZ**2) / (2 * np.pi)).astype(int)
        _CACHE[N] = dict(KX=KX, KY=KY, KZ=KZ, K2=K2, KBIN=KBIN,
                         NBIN=int(KBIN.max()) + 1)
    return _CACHE[N]


def kbin(N):  return grids(N)["KBIN"]
def nbin(N):  return grids(N)["NBIN"]
def kphys(N): return np.arange(grids(N)["NBIN"]) * 2 * np.pi


def ddx(f, axis):
    g = grids(f.shape[0])
    K = (g["KX"], g["KY"], g["KZ"])[axis]
    return np.fft.ifftn(1j * K * np.fft.fftn(f)).real


def load_xyz(p):
    """Load primitives, transposing HDF5 [z,y,x] -> [x,y,z]."""
    with h5py.File(p, "r") as f:
        T = lambda a: np.transpose(np.asarray(a, np.float64), (2, 1, 0))
        return (float(f["time"][0]), T(f["density"]),
                T(f["velocity_x"]), T(f["velocity_y"]), T(f["velocity_z"]))


def helmholtz(u, v, w):
    g = grids(u.shape[0])
    KX, KY, KZ, K2 = g["KX"], g["KY"], g["KZ"], g["K2"]
    U = [u - u.mean(), v - v.mean(), w - w.mean()]
    Uh = [np.fft.fftn(c) for c in U]
    proj = (KX * Uh[0] + KY * Uh[1] + KZ * Uh[2]) / K2
    ud = [np.fft.ifftn(KX * proj).real, np.fft.ifftn(KY * proj).real,
          np.fft.ifftn(KZ * proj).real]
    us = [U[i] - ud[i] for i in range(3)]
    return us, ud


def advect(u, v, w, target):
    """-(u.grad) target_i for each component i."""
    return [-(u * ddx(target[i], 0) + v * ddx(target[i], 1)
             + w * ddx(target[i], 2)) for i in range(3)]


def shell_transfer(probe, A, N):
    """(1/N^6) Re[ probe_hat* . A_hat ] binned by |k| shell."""
    KB = kbin(N); nb = nbin(N)
    T = np.zeros(nb)
    for i in range(3):
        ph = np.fft.fftn(probe[i]); Ah = np.fft.fftn(A[i])
        contrib = np.real(np.conj(ph) * Ah) / N**6
        T += np.bincount(KB.ravel(), weights=contrib.ravel(), minlength=nb)[:nb]
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="out_blast_128_mid_seed1")
    ap.add_argument("--name", default="blast_128_mid_seed1")
    ap.add_argument("--t0", type=float, default=0.08)
    ap.add_argument("--t1", type=float, default=0.115)
    ap.add_argument("--label", default=None, help="output tag (default: from N)")
    args = ap.parse_args()

    snaps = sorted(p for p in glob.glob(str(run_dir(args).run / f"{args.name}_*.h5"))
                   if not p.endswith(".ckpt.h5") and not p.endswith("_spectra.h5"))
    sel = []
    for p in snaps:
        with h5py.File(p, "r") as f:
            t = float(f["time"][0])
        if args.t0 <= t <= args.t1:
            sel.append((t, p))
    if not sel:
        print(f"no snapshots in [{args.t0},{args.t1}] of {args.run}", file=sys.stderr)
        return 1
    N = h5py.File(sel[0][1], "r")["density"].shape[0]
    label = args.label or f"{N}"
    print(f"{len(sel)} snapshots in [{args.t0},{args.t1}], N={N}")

    nb = nbin(N)
    Tds = np.zeros(nb); Tsd = np.zeros(nb)
    Sds = Ssd = Ksol = Kdil = 0.0
    for t, p in sel:
        _, rho, u, v, w = load_xyz(p)
        us, ud = helmholtz(u, v, w)
        U = [us[i] + ud[i] for i in range(3)]
        A = advect(U[0], U[1], U[2], ud)
        B = advect(U[0], U[1], U[2], us)
        Sds += sum(np.mean(us[i] * A[i]) for i in range(3))
        Ssd += sum(np.mean(ud[i] * B[i]) for i in range(3))
        Tds += shell_transfer(us, A, N)
        Tsd += shell_transfer(ud, B, N)
        ks = 0.5 * np.mean(sum(us[i]**2 for i in range(3)))
        kd = 0.5 * np.mean(sum(ud[i]**2 for i in range(3)))
        Ksol += ks; Kdil += kd
        print(f"  t={t:.4f}  Sigma_d->s={sum(np.mean(us[i]*A[i]) for i in range(3)):+.3e}  "
              f"K_sol/K_dil={ks/kd:.3f}")
    n = len(sel)
    Tds /= n; Tsd /= n; Sds /= n; Ssd /= n; Ksol /= n; Kdil /= n
    k = kphys(N)
    print(f"\n  N={N} window-avg K_sol/K_dil={Ksol/Kdil:.3f}")
    print(f"  GLOBAL Sigma_d->s={Sds:+.4e}  Sigma_s->d={Ssd:+.4e}  "
          f"(sum T_d->s={Tds.sum():+.4e})")
    print(f"  net dil->sol: {'POSITIVE' if Sds>0 else 'NEGATIVE'}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(k[1:], Tds[1:], lw=2, color="tab:green", label=r"$T_{d\to s}(k)$")
    ax[0].plot(k[1:], Tsd[1:], lw=2, color="tab:orange", label=r"$T_{s\to d}(k)$")
    ax[0].axhline(0, color="k", lw=0.7); ax[0].set_xscale("log")
    ax[0].set_xlabel("k"); ax[0].set_ylabel("inter-mode transfer")
    ax[0].set_title("scale-resolved nonlinear transfer")
    ax[0].legend(frameon=False); ax[0].grid(True, which="both", ls=":", alpha=0.4)
    ax[1].plot(k[1:], np.cumsum(Tds[1:]), lw=2, color="tab:green",
               label=r"cumulative $T_{d\to s}$")
    ax[1].axhline(Sds, color="tab:green", ls=":", lw=1, label=r"global $\Sigma_{d\to s}$")
    ax[1].axhline(0, color="k", lw=0.7); ax[1].set_xscale("log")
    ax[1].set_xlabel("k"); ax[1].set_ylabel("cumulative transfer")
    ax[1].set_title("cumulative dil->sol"); ax[1].legend(frameon=False)
    ax[1].grid(True, which="both", ls=":", alpha=0.4)
    fig.suptitle(f"Dil<->sol nonlinear transfer ({args.run}, N={N})", y=1.02)
    fig.tight_layout()
    out = ROOT / "figs" / f"intermode_transfer_{label}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")
    np.savez(DATA / f"intermode_transfer_{label}.npz", k=k, T_d_to_s=Tds,
             T_s_to_d=Tsd, Sigma_d_to_s=Sds, Sigma_s_to_d=Ssd,
             K_sol=Ksol, K_dil=Kdil, N=N)
    return 0


if __name__ == "__main__":
    sys.exit(main())
