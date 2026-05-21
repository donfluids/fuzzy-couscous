#!/usr/bin/env python3
"""Does the compressible/dilatational field feed the solenoidal (vortical) field?
(N-agnostic; --case 128 = budget seed1, 256 = match128.)

A blast IC is purely radial = dilatational, ~zero initial vorticity, so any
solenoidal energy is GENERATED from the compressible/thermodynamic field.

ROBUST diagnostics (finite-difference, local; baroclinic production & enstrophy
are axis-permutation-invariant scalars so the HDF5 [z,y,x] order doesn't matter):
  - Baroclinic enstrophy production  B = < omega . (grad rho x grad p)/rho^2 >.
  - Enstrophy <omega^2>.
K_sol/K_dil(t) from the solver spectra (summed bins).

Outputs figs/baroclinic_source_<case>.png, baroclinic_source_<case>.npz.
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
NB = 3


def case_run(case):
    if case == "128":
        return "out_blast_128_budget_seed1", "blast_128_budget_seed1"
    return "out_blast_256_match128", "blast_256_match128"


def load(p):
    with h5py.File(p, "r") as f:
        g = lambda k: np.asarray(f[k], np.float64)
        return (float(f["time"][0]), g("density"), g("velocity_x"),
                g("velocity_y"), g("velocity_z"), g("pressure"))


def snapshots(run_dir, run_name):
    pat = re.compile(rf"{re.escape(run_name)}_(\d+)\.h5$")
    return [p for p in sorted(run_dir.glob(f"{run_name}_*.h5"))
            if pat.search(p.name) and not p.name.endswith(".ckpt.h5")
            and not p.name.endswith("_spectra.h5")]


def solver_KsolKdil(spec_path):
    out = {}
    with h5py.File(spec_path, "r") as f:
        for key in f.keys():
            g = f[key]
            out[float(g["time"][0])] = (np.asarray(g["E_sol"]).sum(),
                                        np.asarray(g["E_dil"]).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="256", choices=["128", "256"])
    args = ap.parse_args()
    run, name = case_run(args.case)
    rd = ROOT / run
    snaps = snapshots(rd, name)
    if not snaps:
        print("no snapshots", file=sys.stderr); return 1
    ks = solver_KsolKdil(rd / f"{name}_spectra.h5")
    sk_t = np.array(sorted(ks)); Ksol_s = np.array([ks[t][0] for t in sk_t])
    Kdil_s = np.array([ks[t][1] for t in sk_t])

    t, B, Ens = [], [], []
    for p in snaps:
        tt, rho, u, v, w, pr = load(p)
        N = rho.shape[0]; dx = 1.0 / N
        gr = lambda f: np.gradient(f, dx, edge_order=2)
        ux, uy, uz = gr(u); vx, vy, vz = gr(v); wx, wy, wz = gr(w)
        ox, oy, oz = wy - vz, uz - wx, vx - uy
        rx, ry, rz = gr(rho); px, py, pz = gr(pr)
        r2 = rho * rho
        bx = (ry*pz - rz*py)/r2; by = (rz*px - rx*pz)/r2; bz = (rx*py - ry*px)/r2
        sl = (slice(NB, -NB),) * 3
        t.append(tt)
        B.append(float((ox*bx + oy*by + oz*bz)[sl].mean()))
        Ens.append(float((ox*ox + oy*oy + oz*oz)[sl].mean()))
        print(f"  t={tt:.3f}  baroclinic={B[-1]:+.3e}  <omega^2>={Ens[-1]:.3e}")
    t = np.array(t); B = np.array(B); Ens = np.array(Ens)
    o = np.argsort(t); t, B, Ens = t[o], B[o], Ens[o]

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ax[0].plot(sk_t, Ksol_s, lw=2, color="tab:blue", label=r"$K_{sol}$")
    ax[0].plot(sk_t, Kdil_s, lw=2, color="tab:red", label=r"$K_{dil}$")
    ax[0].set_yscale("log"); ax[0].set_xlabel("t"); ax[0].set_ylabel("KE")
    ax[0].set_title("Helmholtz KE (solver)"); ax[0].legend(frameon=False)
    ax[0].grid(True, which="both", ls=":", alpha=0.4)
    ax[1].plot(t, B, lw=2, color="tab:purple"); ax[1].axhline(0, color="k", lw=0.7)
    ax[1].set_xlabel("t"); ax[1].set_ylabel(r"$\langle\omega\cdot(\nabla\rho\times\nabla p)/\rho^2\rangle$")
    ax[1].set_title("baroclinic enstrophy production"); ax[1].grid(True, ls=":", alpha=0.4)
    ax[2].plot(t, Ens, lw=2, color="tab:green"); ax[2].set_xlabel("t")
    ax[2].set_ylabel(r"$\langle\omega^2\rangle$"); ax[2].set_title("enstrophy")
    ax[2].grid(True, ls=":", alpha=0.4)
    fig.suptitle(f"Compressible -> vortical generation, case {args.case}", y=1.02)
    fig.tight_layout()
    out = ROOT / "figs" / f"baroclinic_source_{args.case}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); print(f"wrote {out}")
    np.savez(ROOT / f"baroclinic_source_{args.case}.npz", t=t, baroclinic=B,
             enstrophy=Ens, sk_t=sk_t, K_sol=Ksol_s, K_dil=Kdil_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
