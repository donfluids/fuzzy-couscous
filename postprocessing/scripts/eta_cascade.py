#!/usr/bin/env python3
"""Kolmogorov scale eta(t) from the dissipation anomaly (Taylor's cascade law).

Why a global eta(t), not a local eta/dx map: in this blast LES the dissipation
that *defines* eta is dominated by sub-grid numerical dissipation and reversible
compressible exchange -- neither measurable per cell. Every measurable local
estimate either under-counts (resolved viscous / small-r structure function) or
is contaminated (resolved TKE loss -dk/dt mixes in pressure-dilatation). So we
estimate eta GLOBALLY from the resolved energy-containing scales via the
dissipation anomaly (Taylor 1935 / "zeroth law"):

    eps = C_eps * u'^3 / L          (cascade rate set by the large scales)
    eta = (nu^3 / eps)^(1/4),  nu = mu/rho_mean
    cross-check:  eta/L = Re_L^(-3/4),  Re_L = u' L / nu.

u' and L come from the resolved large scales (reliable), so eps is the true
rate the cascade must carry regardless of whether the grid resolves the
dissipation range. We compute eta as a function of time.

Solenoidal vs total. In this flow most resolved fluctuation energy is
dilatational/acoustic (f_sol ~ 0.1-0.3). The vortical cascade is the solenoidal
part, so we report eta from the SOLENOIDAL energy as primary, and the
TOTAL-energy cascade as an upper bound on turbulent intensity. We also overlay
the optimistic resolved-viscous eta and the (compressibility-contaminated)
throughput eta so the spread is explicit.

Note on RANS: a 1D spherical BHR cannot supply a "true eps" here -- in spherical
symmetry grad(rho) || grad(p), so the baroclinic torque that produces the BHR
mass flux a_i and covariance b vanishes; its eps is not self-consistent. A
trustworthy model eps would need a multi-dimensional BHR. Hence eta here rests
on the resolved cascade, not a 1D model.

Outputs: figs/eta_cascade.png and eta_cascade.npz.
"""

import argparse
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

MU = 5.0e-4
N = 128
L_BOX = 1.0
DX = L_BOX / N

SEEDS = [1, 2, 3, 4, 5]
TAG = "budget"            # "budget" (t<=0.5) or "early" (dense t<=0.02)
def RUN_DIR(s): return run_dir(f"out_blast_128_{TAG}_seed{s}")
def RUN_NAME(s): return f"blast_128_{TAG}_seed{s}"

C_EPS = 0.5            # dissipation-anomaly constant (eps = C_eps u'^3 / L)


def load_stats(path):
    cols = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return {k: np.array(v) for k, v in cols.items()}


def spectra_integral_scale(path):
    """Per spectra step: time, longitudinal integral scale L (from E_sol), and
    solenoidal energy fraction f_sol = K_sol/K_tot."""
    t, L, fsol = [], [], []
    with h5py.File(path, "r") as f:
        for key in f.keys():
            g = f[key]
            k = np.asarray(g["k"], np.float64)
            Es = np.asarray(g["E_sol"], np.float64)
            Et = np.asarray(g["E_total"], np.float64)
            tt = float(g["time"][0])
            m = k > 0
            Ksol = np.trapz(Es[m], k[m])
            Ktot = np.trapz(Et[m], k[m])
            if Ksol <= 0 or Ktot <= 0:
                continue
            Lint = (3.0 * np.pi / 4.0) * np.trapz(Es[m] / k[m], k[m]) / Ksol
            t.append(tt); L.append(Lint); fsol.append(Ksol / Ktot)
    o = np.argsort(t)
    return np.array(t)[o], np.array(L)[o], np.array(fsol)[o]


def eta_of(nu, eps):
    return np.power(np.power(nu, 3) / np.maximum(eps, 1e-300), 0.25)


def seed_estimates(seed, c_eps):
    st = load_stats(RUN_DIR(seed) / f"{RUN_NAME(seed)}_stats.csv")
    t = st["time"]
    rho_m = st["rho_mean"]
    nu = MU / rho_m
    k_mass = st["tke"] / rho_m                          # TKE per unit mass
    up_tot = np.sqrt(np.maximum(2.0 * k_mass / 3.0, 0.0))  # one-comp rms (total)

    ts, Ls, fs = spectra_integral_scale(
        RUN_DIR(seed) / f"{RUN_NAME(seed)}_spectra.h5")
    L = np.interp(t, ts, Ls)
    fsol = np.clip(np.interp(t, ts, fs), 0, 1)
    up_sol = up_tot * np.sqrt(fsol)                     # solenoidal one-comp rms

    eps_sol = c_eps * up_sol ** 3 / np.maximum(L, 1e-30)
    eps_tot = c_eps * up_tot ** 3 / np.maximum(L, 1e-30)
    eps_visc = st["eps_total"]                          # resolved viscous
    eps_thru = -np.gradient(k_mass, t)                  # throughput
    eps_thru = np.where(eps_thru > 0, eps_thru, np.nan)  # mask reverse exchange

    return dict(
        t=t, nu=nu, L=L, fsol=fsol, up_sol=up_sol, up_tot=up_tot,
        ReL_sol=up_sol * L / nu, ReL_tot=up_tot * L / nu,
        eta_sol=eta_of(nu, eps_sol), eta_tot=eta_of(nu, eps_tot),
        eta_visc=eta_of(nu, eps_visc), eta_thru=eta_of(nu, eps_thru),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--C-eps", type=float, default=C_EPS)
    ap.add_argument("--single", action="store_true",
                    help="use seed 1 only (default: ensemble over available seeds)")
    ap.add_argument("--tag", default="budget",
                    help="run set: 'budget' (t<=0.5) or 'early' (dense t<=0.02)")
    args = ap.parse_args()

    global TAG
    TAG = args.tag
    figs = ROOT / "figs"; figs.mkdir(parents=True, exist_ok=True)

    seeds = [s for s in SEEDS
             if (RUN_DIR(s) / f"{RUN_NAME(s)}_stats.csv").exists()
             and (RUN_DIR(s) / f"{RUN_NAME(s)}_spectra.h5").exists()]
    if not seeds:
        print("No seed with both stats and spectra.", file=sys.stderr)
        return 1
    use = seeds[:1] if args.single else seeds
    print(f"seeds {use}  C_eps={args.C_eps}")

    tg = seed_estimates(use[0], args.C_eps)["t"]

    def avg(field):
        acc = np.zeros_like(tg)
        for s in use:
            e = seed_estimates(s, args.C_eps)
            acc += np.interp(tg, e["t"], e[field])
        return acc / len(use)

    nu = avg("nu"); L = avg("L"); fsol = avg("fsol")
    ReL_sol = avg("ReL_sol"); ReL_tot = avg("ReL_tot")
    eta_sol = avg("eta_sol"); eta_tot = avg("eta_tot")
    eta_visc = avg("eta_visc"); eta_thru = avg("eta_thru")

    # printed table -- eta as a function of time
    if tg.max() < 0.05:        # early-time dense run
        table_t = (0.001, 0.002, 0.004, 0.006, 0.008, 0.010, 0.014, 0.018)
    else:
        table_t = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
    print("\n  t     eta_sol   eta_tot   eta_visc   dx       "
          "eta/dx[sol]  eta/dx[tot]  Re_L[sol]  f_sol")
    for tt in table_t:
        i = int(np.argmin(np.abs(tg - tt)))
        print(f"  {tg[i]:.3f}  {eta_sol[i]:.2e}  {eta_tot[i]:.2e}  "
              f"{eta_visc[i]:.2e}  {DX:.2e}  {eta_sol[i]/DX:9.3f}    "
              f"{eta_tot[i]/DX:9.3f}    {ReL_sol[i]:7.0f}   {fsol[i]:.3f}")

    # ---------------------------------------------------------------- plots
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    a = ax[0, 0]
    a.plot(tg, eta_sol, lw=2.4, color="tab:blue", label="solenoidal cascade (primary)")
    a.plot(tg, eta_tot, lw=2.0, color="tab:cyan", label="total-energy cascade")
    a.plot(tg, eta_visc, lw=1.6, color="tab:orange", label="resolved viscous (optimistic)")
    a.plot(tg, eta_thru, lw=1.4, color="tab:green", label=r"throughput $-dk/dt$")
    a.axhline(DX, color="k", ls="--", lw=1.2, label=r"$\Delta x$")
    a.set_yscale("log"); a.set_xlabel("t"); a.set_ylabel(r"$\eta$  (absolute)")
    a.set_title(r"Kolmogorov scale $\eta(t)$")
    a.legend(frameon=False, fontsize=8); a.grid(True, which="both", ls=":", alpha=0.4)

    a = ax[0, 1]
    a.plot(tg, eta_sol / DX, lw=2.4, color="tab:blue", label="solenoidal cascade")
    a.plot(tg, eta_tot / DX, lw=2.0, color="tab:cyan", label="total-energy cascade")
    a.plot(tg, eta_visc / DX, lw=1.6, color="tab:orange", label="resolved viscous")
    a.axhline(1.0, color="k", ls="--", lw=1, alpha=0.6)
    a.axhline(0.5, color="red", ls=":", lw=1, alpha=0.6)
    a.set_yscale("log"); a.set_xlabel("t"); a.set_ylabel(r"$\eta/\Delta x$")
    a.set_title(r"$\eta/\Delta x(t)$   (>1 grid-resolves $\eta$)")
    a.legend(frameon=False, fontsize=8); a.grid(True, which="both", ls=":", alpha=0.4)

    a = ax[1, 0]
    a.plot(tg, ReL_sol, lw=2.4, color="tab:blue", label=r"$Re_L$ solenoidal")
    a.plot(tg, ReL_tot, lw=2.0, color="tab:cyan", label=r"$Re_L$ total")
    a.set_xlabel("t"); a.set_ylabel(r"$Re_L = u' L/\nu$")
    a.set_title("integral-scale Reynolds number")
    a.legend(frameon=False, fontsize=8); a.grid(True, ls=":", alpha=0.4)

    a = ax[1, 1]
    a.plot(tg, fsol, lw=2.4, color="tab:purple")
    a.set_xlabel("t"); a.set_ylabel(r"solenoidal fraction $K_{sol}/K_{tot}$",
                                    color="tab:purple")
    a.set_ylim(0, 1); a.grid(True, ls=":", alpha=0.4)
    a.set_title("vortical vs acoustic energy split")
    ab = a.twinx()
    ab.plot(tg, L, lw=1.6, color="tab:gray", alpha=0.7)
    ab.set_ylabel(r"integral scale $L$", color="tab:gray")

    fig.suptitle(f"Kolmogorov scale via dissipation anomaly  "
                 f"(seeds {use}, $C_\\varepsilon$={args.C_eps})", y=1.0)
    fig.tight_layout()
    out = figs / f"eta_cascade_{TAG}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out}")

    np.savez(DATA / f"eta_cascade_{TAG}.npz",
             t=tg, eta_sol=eta_sol, eta_tot=eta_tot, eta_visc=eta_visc,
             eta_thru=eta_thru, eta_dx_sol=eta_sol / DX, eta_dx_tot=eta_tot / DX,
             nu=nu, L=L, fsol=fsol, ReL_sol=ReL_sol, ReL_tot=ReL_tot,
             dx=DX, C_eps=args.C_eps, seeds=np.array(use))
    print(f"wrote {DATA/('eta_cascade_'+TAG+'.npz')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
