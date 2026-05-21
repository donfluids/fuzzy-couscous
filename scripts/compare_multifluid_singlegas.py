#!/usr/bin/env python3
"""Does a persistent dense-products / light-air material interface generate
STRONGER, more SUSTAINED baroclinic vorticity than an equivalent single-gas blast?

Controlled comparison (identical IC except the fluid model):
  - MF  : two-gamma multifluid. Products = Chapman-Jouguet state (rho_CJ, p_CJ),
          gamma_p=1.25 (denser, larger cv); air gamma=1.4. G=1/(gamma-1) advected,
          so the material/gamma interface stays SHARP and PERSISTENT.
  - SG  : single gas, gamma=1.4 everywhere, blob initialised at the SAME (rho, p, T)
          as the CJ products. Same r0, same Y42 seed, same hyper/visc.
The only difference is the fluid model -> isolates the effect of the persistent
variable-gamma material interface on baroclinic vorticity (grad rho x grad p).

Diagnostics per snapshot (FD; baroclinic production & enstrophy are
axis-permutation-invariant scalars so the HDF5 [z,y,x] order is irrelevant):
  - baroclinic enstrophy production  B = < omega . (grad rho x grad p)/rho^2 >
  - enstrophy  <omega^2>
  - solenoidal fraction of KE from the solver Helmholtz spectra (summed bins)

Outputs figs/compare_multifluid_singlegas.png + .npz.
"""
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUNROOT = ROOT / "solver"   # blast_les writes out_dir relative to solver/
NB = 3  # trim border cells (one-sided FD / slip-wall artefacts)

CASES = {
    "MF (two-gamma, persistent interface)": ("out_blast_128_cj_t5", "blast_128_cj_t5", "tab:red"),
    "SG (single gas, gamma=1.4)":           ("out_blast_128_sg_t5", "blast_128_sg_t5", "tab:blue"),
}


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
    if not spec_path.exists():
        return out
    with h5py.File(spec_path, "r") as f:
        for key in f.keys():
            g = f[key]
            out[float(g["time"][0])] = (float(np.asarray(g["E_sol"]).sum()),
                                        float(np.asarray(g["E_dil"]).sum()))
    return out


def analyse(run, name):
    rd = RUNROOT / run
    snaps = snapshots(rd, name)
    t, B, Ens, Ksol_fd = [], [], [], []
    for p in snaps:
        tt, rho, u, v, w, pr = load(p)
        N = rho.shape[0]
        dx = 1.0 / N
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
    t = np.array(t); B = np.array(B); Ens = np.array(Ens)
    o = np.argsort(t); t, B, Ens = t[o], B[o], Ens[o]

    ks = solver_KsolKdil(rd / f"{name}_spectra.h5")
    sk_t = np.array(sorted(ks))
    Ksol = np.array([ks[x][0] for x in sk_t]) if len(sk_t) else np.array([])
    Kdil = np.array([ks[x][1] for x in sk_t]) if len(sk_t) else np.array([])
    fsol = Ksol / (Ksol + Kdil + 1e-300) if len(sk_t) else np.array([])
    return dict(t=t, B=B, Ens=Ens, sk_t=sk_t, Ksol=Ksol, Kdil=Kdil, fsol=fsol)


def main():
    res = {}
    for label, (run, name, _c) in CASES.items():
        if not (RUNROOT / run).exists():
            print(f"missing {run}", file=sys.stderr)
            continue
        res[label] = analyse(run, name)
        r = res[label]
        print(f"{label}: {len(r['t'])} snaps; "
              f"enstrophy max={r['Ens'].max() if len(r['Ens']) else 0:.3e}; "
              f"f_sol max={r['fsol'].max() if len(r['fsol']) else 0:.3f}")

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for label, (run, name, col) in CASES.items():
        if label not in res:
            continue
        r = res[label]
        ax[0, 0].plot(r["t"], r["Ens"], lw=2, color=col, marker="o", ms=3, label=label)
        ax[0, 1].plot(r["t"], r["B"], lw=2, color=col, marker="o", ms=3, label=label)
        if len(r["sk_t"]):
            ax[1, 0].plot(r["sk_t"], r["Ksol"], lw=2, color=col, label=label + r" $K_{sol}$")
            ax[1, 0].plot(r["sk_t"], r["Kdil"], lw=1.2, ls="--", color=col, alpha=0.6,
                          label=label + r" $K_{dil}$")
            ax[1, 1].plot(r["sk_t"], r["fsol"], lw=2, color=col, marker="o", ms=3, label=label)

    ax[0, 0].set_title(r"Enstrophy $\langle\omega^2\rangle$ (vortical content)")
    ax[0, 0].set_xlabel("t"); ax[0, 0].set_yscale("log")
    ax[0, 1].axhline(0, color="k", lw=0.7)
    ax[0, 1].set_title(r"Baroclinic production $\langle\omega\cdot(\nabla\rho\times\nabla p)/\rho^2\rangle$")
    ax[0, 1].set_xlabel("t")
    ax[1, 0].set_title("Helmholtz KE (solver spectra)")
    ax[1, 0].set_xlabel("t"); ax[1, 0].set_yscale("log")
    ax[1, 1].set_title(r"Solenoidal fraction $K_{sol}/(K_{sol}+K_{dil})$")
    ax[1, 1].set_xlabel("t")
    for a in ax.flat:
        a.grid(True, which="both", ls=":", alpha=0.4)
        a.legend(frameon=False, fontsize=8)
    fig.suptitle("Persistent dense-products interface vs single-gas blast (128$^3$)", y=1.0)
    fig.tight_layout()
    out = ROOT / "figs" / "compare_multifluid_singlegas.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")
    np.savez(ROOT / "compare_multifluid_singlegas.npz",
             **{f"{lab.split()[0]}_{k}": v for lab, r in res.items() for k, v in r.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
