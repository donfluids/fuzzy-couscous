#!/usr/bin/env python3
"""BHR k-equation budget from the 128^3 blast ensemble (5 seeds, frequent snapshots).

For each snapshot time available across all 5 ensemble members, compute the
ensemble-averaged budget of the Reynolds-averaged turbulent kinetic energy:

    d<rho>k/dt   ~=  P_shear  +  P_VD  -  <rho>*eps  +  residual

  P_shear = -<rho> R_ij dU_i/dx_j        (mean-shear production)
  P_VD    = -a_i dp_mean/dx_i             (BHR variable-density production)
  eps     = (taken from per-snapshot pointwise estimate via stats CSV)

  residual = pressure-dilatation + turbulent transport + closure terms not modeled

All terms volume-averaged over the chamber. Plot:
  - <k>(t) and its time derivative
  - each budget term vs t
  - residual vs t (= Pi_d + transport, what BHR k-eq closure must approximate)

Tests the BHR ansatz: is P_VD the dominant source? Is the standard k-eps
production P_shear negligible? How big is the residual that BHR's pressure-
dilatation and transport closures would need to cover?
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
GAMMA = 1.4
MU = 5.0e-4         # physical viscosity (matches blast_128_budget_*.toml)
NU6 = 2.5e-14       # hyperdissipation coefficient

SEEDS = [1, 2, 3, 4, 5]
RUN_DIR = lambda s: run_dir(f"out_blast_128_budget_seed{s}")
RUN_NAME = lambda s: f"blast_128_budget_seed{s}"


def list_snapshots(run_dir, run_name):
    """Return sorted (step, time, path) for each snapshot HDF5 in run_dir.
    Snapshots store primitive vars; step is parsed from the filename
    suffix (run_name_NNNNNN.h5)."""
    import re
    files = sorted(run_dir.glob(f"{run_name}_*.h5"))
    out = []
    pat = re.compile(rf"{re.escape(run_name)}_(\d+)\.h5$")
    for p in files:
        if p.name.endswith(".ckpt.h5"):
            continue
        m = pat.search(p.name)
        if not m:
            continue
        step = int(m.group(1))
        with h5py.File(p, "r") as f:
            t = float(f["time"][0]) if "time" in f else float("nan")
        out.append((step, t, p))
    return out


def load_state(p):
    """Snapshots store primitive variables directly."""
    with h5py.File(p, "r") as f:
        rho = np.array(f["density"],     dtype=np.float64)
        u   = np.array(f["velocity_x"],  dtype=np.float64)
        v   = np.array(f["velocity_y"],  dtype=np.float64)
        w   = np.array(f["velocity_z"],  dtype=np.float64)
        p_  = np.array(f["pressure"],    dtype=np.float64)
        t = float(f["time"][0])
    return t, rho, u, v, w, p_


def cross_seed_table(seeds):
    """Build a table of (step -> {seed: path}) so we can pick snapshot
    sets that exist across all ensemble members."""
    tables = {}
    for s in seeds:
        snaps = list_snapshots(RUN_DIR(s), RUN_NAME(s))
        for step, t, p in snaps:
            tables.setdefault(step, {})[s] = (t, p)
    # Keep only steps present in every seed.
    full = {step: d for step, d in tables.items() if len(d) == len(seeds)}
    return full


def gradient(f, h):
    """Central-diff gradient (axis-by-axis) in periodic-style; slip-wall
    boundaries use one-sided diffs at the edges (numpy default)."""
    return (np.gradient(f, h, axis=0),
            np.gradient(f, h, axis=1),
            np.gradient(f, h, axis=2))


def eps_hyper_from_snapshot(u, v, w, h, nu6, rho_mean=1.0):
    """Effective hyperdissipation rate per unit volume:
       eps_hyper ~ nu6 * rho * < (d^3 u / dx^3)^2 + ... >
    using FD third derivatives. Incompressible-like approximation (treats
    the hyperdiss operator on rho u_i as if rho were locally constant);
    accurate to O(rho'/rho) in our chamber where rho'/rho ~ 0.2.

    Returns volume-averaged scalar.
    """
    def lap(f):
        return (np.gradient(np.gradient(f, h, axis=0), h, axis=0)
              + np.gradient(np.gradient(f, h, axis=1), h, axis=1)
              + np.gradient(np.gradient(f, h, axis=2), h, axis=2))

    total = 0.0
    for f in (u, v, w):
        l1 = lap(f)
        l2 = lap(l1)
        l3 = lap(l2)          # (nabla^2)^3 f, equals -(-nabla^2)^3 f
        # Hyperdiss term in our solver is +nu6 (nabla^2)^3 U (sign +).
        # KE contribution per cell: u_i * d(rho u_i)/dt |_hyper
        # In incompressible-like limit: u_i * nu6 (nabla^2)^3 (rho u_i)
        # ~ rho u_i * nu6 (nabla^2)^3 u_i (treat rho ~ const).
        # Integrated KE rate: -nu6 * rho * <|nabla^3 u_i|^2> (IBP, sign convention).
        # We report eps_hyper as a positive number (dissipation rate).
        total += float(np.mean(f * l3))
    # < f * (nabla^2)^3 f > by IBP = -<|nabla^3 f|^2> for periodic; the sign
    # below makes eps_hyper positive (a sink in the KE budget).
    eps_hyper = -nu6 * rho_mean * total
    return eps_hyper


def main():
    full = cross_seed_table(SEEDS)
    if not full:
        print("No snapshots present across all ensemble members yet.",
              file=sys.stderr)
        return 1

    steps = sorted(full.keys())
    print(f"Found {len(steps)} common snapshot steps "
          f"across {len(SEEDS)} ensemble members")
    print(f"Step range: {steps[0]} ... {steps[-1]}")

    nx = 128
    h = 1.0 / nx
    V = 1.0  # box volume

    times = []
    K_int     = []
    P_shear   = []
    P_VD      = []
    PiD       = []
    Eps_visc  = []      # classical viscous part (from stats CSV)
    Eps_hyper = []      # NEW: hyperdissipative part (from snapshot derivatives)

    a_mag_int = []
    b_int     = []
    rho_var_int = []

    for step in steps:
        rhos, us, vs, ws, ps = [], [], [], [], []
        t_avg = 0.0
        for s in SEEDS:
            ts, rho, u, v, w, p = load_state(full[step][s][1])
            rhos.append(rho); us.append(u); vs.append(v)
            ws.append(w); ps.append(p)
            t_avg += ts
        t_avg /= len(SEEDS)
        times.append(t_avg)

        rhos = np.stack(rhos); us = np.stack(us); vs = np.stack(vs)
        ws = np.stack(ws); ps = np.stack(ps)

        # Reynolds averages (pointwise across ensemble).
        rho_m = rhos.mean(0); u_m = us.mean(0); v_m = vs.mean(0)
        w_m   = ws.mean(0);   p_m = ps.mean(0)
        rho_p = rhos - rho_m
        u_p   = us   - u_m;   v_p = vs - v_m;   w_p = ws - w_m

        # k and Reynolds stresses.
        R_uu = (u_p * u_p).mean(0)
        R_uv = (u_p * v_p).mean(0)
        R_uw = (u_p * w_p).mean(0)
        R_vv = (v_p * v_p).mean(0)
        R_vw = (v_p * w_p).mean(0)
        R_ww = (w_p * w_p).mean(0)
        k_field = 0.5 * (R_uu + R_vv + R_ww)

        # Mass flux a_i.
        a_x = (rho_p * u_p).mean(0) / rho_m
        a_y = (rho_p * v_p).mean(0) / rho_m
        a_z = (rho_p * w_p).mean(0) / rho_m

        # Density-spec-vol covariance b.
        v_inv_m = (1.0 / rhos).mean(0)
        b_field = rho_m * v_inv_m - 1.0

        # Mean velocity gradients.
        dUx_dx, dUx_dy, dUx_dz = gradient(u_m, h)
        dUy_dx, dUy_dy, dUy_dz = gradient(v_m, h)
        dUz_dx, dUz_dy, dUz_dz = gradient(w_m, h)

        # Mean pressure gradient.
        dpx, dpy, dpz = gradient(p_m, h)

        # Mean-shear production: -<rho> R_ij dU_i/dx_j.
        P_shear_field = -rho_m * (
            R_uu*dUx_dx + R_uv*(dUx_dy + dUy_dx) + R_uw*(dUx_dz + dUz_dx)
            + R_vv*dUy_dy + R_vw*(dUy_dz + dUz_dy) + R_ww*dUz_dz)

        # Variable-density production: -a_i dp_m/dx_i.
        P_VD_field = -(a_x*dpx + a_y*dpy + a_z*dpz)

        # Pressure-dilatation: <p' (du'_i/dx_i)> from ensemble.
        # Use Reynolds fluctuation p' and Reynolds fluctuation dilatation.
        div_u_p = np.zeros_like(rho_m)
        for k_s in range(len(SEEDS)):
            duux, _, _ = gradient(us[k_s] - u_m, h)
            _, duvy, _ = gradient(vs[k_s] - v_m, h)
            _, _, duwz = gradient(ws[k_s] - w_m, h)
            div_u_p_s = duux + duvy + duwz
            div_u_p += (ps[k_s] - p_m) * div_u_p_s
        PiD_field = div_u_p / len(SEEDS)

        # Volume-average each.
        K_int.append(float(np.mean(rho_m * k_field) * V))   # <rho k> volume
        P_shear.append(float(np.mean(P_shear_field) * V))
        P_VD.append(float(np.mean(P_VD_field) * V))
        PiD.append(float(np.mean(PiD_field) * V))

        # Effective hyperdissipation rate, ensemble-averaged.
        eps_h = 0.0
        for ks in range(len(SEEDS)):
            eps_h += eps_hyper_from_snapshot(
                us[ks], vs[ks], ws[ks], h, NU6,
                rho_mean=float(rho_m.mean()))
        Eps_hyper.append(eps_h / len(SEEDS))

        a_mag_int.append(float(np.sqrt(np.mean(a_x**2 + a_y**2 + a_z**2))))
        b_int.append(float(np.mean(b_field)))
        rho_var_int.append(float(np.mean(rho_p.mean(0)**0)))  # placeholder

        print(f"  step={step:6d} t={t_avg:.3f}  "
              f"K={K_int[-1]:.4e}  P_shear={P_shear[-1]:.4e}  "
              f"P_VD={P_VD[-1]:.4e}  PiD={PiD[-1]:.4e}  "
              f"eps_hyper={Eps_hyper[-1]:.4e}  "
              f"|a|={a_mag_int[-1]:.3e}  b={b_int[-1]:.3e}")

    times = np.array(times)
    K_int   = np.array(K_int)
    P_shear = np.array(P_shear)
    P_VD    = np.array(P_VD)
    PiD     = np.array(PiD)
    Eps_hyper = np.array(Eps_hyper)
    a_mag_int = np.array(a_mag_int)
    b_int     = np.array(b_int)

    # Classical viscous eps from stats CSV (averaged across seeds).
    eps_t_all, eps_v_all = None, None
    for s in SEEDS:
        path = RUN_DIR(s) / f"{RUN_NAME(s)}_stats.csv"
        t_s, e_s = [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                t_s.append(float(row["time"]))
                e_s.append(float(row["eps_total"]))
        t_s = np.array(t_s); e_s = np.array(e_s)
        if eps_t_all is None:
            eps_t_all = t_s
            eps_v_all = e_s
        else:
            # Interpolate this seed onto the first seed's time grid before averaging
            eps_v_all = eps_v_all + np.interp(eps_t_all, t_s, e_s)
    Eps_visc_t = eps_v_all / len(SEEDS)
    Eps_visc   = np.interp(times, eps_t_all, Eps_visc_t)
    Eps_total  = Eps_visc + Eps_hyper

    # Numerical time derivative of <rho k>.
    if len(times) > 1:
        dKdt = np.gradient(K_int, times)
    else:
        dKdt = np.zeros_like(K_int)

    # The k-equation budget (volume-averaged):
    #   d<rho k>/dt = P_shear + P_VD - <rho eps_eff> + Pi_d + transport
    # where eps_eff = eps_visc + eps_hyper covers both physical and LES-SGS dissipation.
    rhs_basic = P_shear + P_VD - Eps_total + PiD
    residual = dKdt - rhs_basic

    # Plots: 3-panel summary.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(times, K_int, lw=1.8, label=r"$\langle\bar\rho k\rangle$")
    ax.plot(times, dKdt, lw=1.6, ls="--",
            label=r"$d\langle\bar\rho k\rangle/dt$")
    ax.set_xlabel(r"$t$")
    ax.set_title(r"ensemble TKE and its time derivative")
    ax.legend(frameon=False)
    ax.grid(True, ls=":", alpha=0.5)

    ax = axes[1]
    ax.plot(times, P_VD,    lw=1.8, color="tab:blue",
            label=r"$P_{\rm VD} = -\langle a_i \partial_i \bar p\rangle$  (BHR)")
    ax.plot(times, P_shear, lw=1.8, color="tab:orange",
            label=r"$P_{\rm shear}$  (standard k-eps)")
    ax.plot(times, -Eps_total, lw=1.8, color="tab:green",
            label=r"$-\varepsilon_{\rm eff} = -(\varepsilon_{\rm visc}+\varepsilon_{\rm hyper})$")
    ax.plot(times, -Eps_visc,  lw=1.0, color="tab:green", ls="--",
            alpha=0.6, label=r"$-\varepsilon_{\rm visc}$ alone")
    ax.plot(times, -Eps_hyper, lw=1.0, color="tab:green", ls=":",
            alpha=0.6, label=r"$-\varepsilon_{\rm hyper}$ alone")
    ax.plot(times, PiD,     lw=1.6, color="tab:purple",
            label=r"$\Pi_d = \langle p' \nabla\!\cdot u''\rangle$")
    ax.plot(times, dKdt,    lw=1.5, ls=":", color="k",
            label=r"$d\langle\bar\rho k\rangle/dt$")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel(r"$t$")
    ax.set_title(r"$k$-budget terms (volume-averaged)")
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.grid(True, ls=":", alpha=0.5)

    ax = axes[2]
    ax.semilogy(times, np.abs(P_VD),    lw=1.8, color="tab:blue",
                label=r"$|P_{\rm VD}|$")
    ax.semilogy(times, np.abs(P_shear), lw=1.8, color="tab:orange",
                label=r"$|P_{\rm shear}|$")
    ax.semilogy(times, np.abs(Eps_total),  lw=1.8, color="tab:green",
                label=r"$|\varepsilon_{\rm eff}|$")
    ax.semilogy(times, np.abs(Eps_hyper),  lw=1.0, color="tab:green",
                ls=":", alpha=0.6,
                label=r"$|\varepsilon_{\rm hyper}|$")
    ax.semilogy(times, np.abs(PiD),     lw=1.6, color="tab:purple",
                label=r"$|\Pi_d|$")
    ax.semilogy(times, np.abs(residual), lw=1.4, ls=":",
                color="tab:red",
                label=r"$|$residual$|$ = transport+closure")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"magnitude")
    ax.set_title("budget magnitudes (log scale)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.5)

    fig.suptitle(
        r"BHR $k$-equation budget on $128^3$ blast ensemble (5 seeds): "
        r"is $P_{\rm VD}$ the dominant source?",
        y=1.02, fontsize=11)
    out = ROOT / "figs" / "bhr_kbudget.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
