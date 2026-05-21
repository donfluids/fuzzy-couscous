#!/usr/bin/env python3
"""A-priori gate #2: the turbulent mass-flux (a_r) and covariance (b) BUDGETS
measured from the 128^3 LES ensemble -- which BHR production term actually
sustains a/b, and does the closure form work?

From the 5-seed ensemble (out_blast_128_budget_seed{1-5}), at each common
snapshot, compute Reynolds averages and the radial mass flux a_r = a.rhat, the
covariance b, Reynolds stresses R_ij, k. Then evaluate the BHR-modeled
production terms (radial component, projected on rhat) and volume-average them
inside the chamber:

  a_r budget:  d<rho a_r>/dt  vs
     P_baro  =  b dp/dx_i                       (variable-density / baroclinic)
     P_Rey   = -(R_ij/rho) drho/dx_j            (Reynolds-stress x density grad)
     P_shear = -rho a_j dU_i/dx_j               (mean-shear)
  b budget:    d<rho b>/dt  vs
     P_b      = -2 a_i drho/dx_i (1+b)

residual = d/dt - sum(productions) = -(dissipation + transport) that BHR closes.

Tells us: (i) which term dominates the DNS a/b generation (the one 1D spherical
can/can't produce), (ii) whether the BHR production forms have the right sign and
magnitude. Reuses the ensemble machinery of scripts/bhr_budget.py.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("bb", ROOT / "scripts" / "bhr_budget.py")
bb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bb)

N = 128
L = 1.0
X0 = -0.5
DX = L / N
R_IN = 0.45            # volume-average inside this radius (avoid wall corners)


def rhat():
    xc = X0 + (np.arange(N) + 0.5) * DX
    X, Y, Z = np.meshgrid(xc, xc, xc, indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z)
    ri = 1.0 / np.maximum(r, 1e-30)
    return r, X * ri, Y * ri, Z * ri


def main():
    full = bb.cross_seed_table(bb.SEEDS)
    if not full:
        print("no ensemble snapshots", file=sys.stderr); return 1
    steps = sorted(full.keys())
    r3, ex, ey, ez = rhat()
    inside = r3 <= R_IN
    Vcell = DX**3
    h = DX

    t_l, dadt_l, db_l = [], [], []
    Pbaro_l, PRey_l, Pshear_l, Pb_l = [], [], [], []
    ar_int_l, b_int_l = [], []

    for step in steps:
        rhos, us, vs, ws, ps = [], [], [], [], []
        tt = 0.0
        for s in bb.SEEDS:
            ts, rho, u, v, w, p = bb.load_state(full[step][s][1])
            rhos.append(rho); us.append(u); vs.append(v); ws.append(w); ps.append(p)
            tt += ts
        tt /= len(bb.SEEDS)
        rhos = np.stack(rhos); us = np.stack(us); vs = np.stack(vs)
        ws = np.stack(ws); ps = np.stack(ps)

        rho_m = rhos.mean(0); u_m = us.mean(0); v_m = vs.mean(0)
        w_m = ws.mean(0); p_m = ps.mean(0)
        rp = rhos - rho_m; up = us - u_m; vp = vs - v_m; wp = ws - w_m

        # mass flux a_i and covariance b
        a_x = (rp * up).mean(0) / rho_m
        a_y = (rp * vp).mean(0) / rho_m
        a_z = (rp * wp).mean(0) / rho_m
        b = rho_m * (1.0 / rhos).mean(0) - 1.0
        # Reynolds stresses
        Ruu = (up*up).mean(0); Ruv = (up*vp).mean(0); Ruw = (up*wp).mean(0)
        Rvv = (vp*vp).mean(0); Rvw = (vp*wp).mean(0); Rww = (wp*wp).mean(0)

        # mean gradients
        dpx, dpy, dpz = bb.gradient(p_m, h)
        drx, dry, drz = bb.gradient(rho_m, h)
        dUxx, dUxy, dUxz = bb.gradient(u_m, h)
        dUyx, dUyy, dUyz = bb.gradient(v_m, h)
        dUzx, dUzy, dUzz = bb.gradient(w_m, h)

        # BHR production terms (vector), then project on rhat and rho-weight
        # P_baro_i = b dp/dx_i
        Pbx = b*dpx; Pby = b*dpy; Pbz = b*dpz
        # P_Rey_i = -(R_ij/rho) drho/dx_j
        PRx = -(Ruu*drx + Ruv*dry + Ruw*drz)/rho_m
        PRy = -(Ruv*drx + Rvv*dry + Rvw*drz)/rho_m
        PRz = -(Ruw*drx + Rvw*dry + Rww*drz)/rho_m
        # P_shear_i = -rho a_j dU_i/dx_j
        PSx = -rho_m*(a_x*dUxx + a_y*dUxy + a_z*dUxz)
        PSy = -rho_m*(a_x*dUyx + a_y*dUyy + a_z*dUyz)
        PSz = -rho_m*(a_x*dUzx + a_y*dUzy + a_z*dUzz)
        # b production = -2 a_i drho/dx_i (1+b)
        Pb = -2.0*(a_x*drx + a_y*dry + a_z*drz)*(1.0+b)

        def vint_r(fx, fy, fz):   # volume integral of radial projection, rho-weighted
            fr = (fx*ex + fy*ey + fz*ez)
            return float((rho_m*fr)[inside].sum()*Vcell)
        def vint(f):
            return float(f[inside].sum()*Vcell)

        ar = a_x*ex + a_y*ey + a_z*ez
        t_l.append(tt)
        ar_int_l.append(float((rho_m*ar)[inside].sum()*Vcell))
        b_int_l.append(float((rho_m*b)[inside].sum()*Vcell))
        Pbaro_l.append(vint_r(Pbx, Pby, Pbz))
        PRey_l.append(vint_r(PRx, PRy, PRz))
        Pshear_l.append(vint_r(PSx, PSy, PSz))
        Pb_l.append(vint(rho_m*Pb))
        print(f"  t={tt:.3f}  <rho a_r>={ar_int_l[-1]:+.3e}  <rho b>={b_int_l[-1]:+.3e}  "
              f"P_baro={Pbaro_l[-1]:+.3e} P_Rey={PRey_l[-1]:+.3e} P_shear={Pshear_l[-1]:+.3e}")

    t = np.array(t_l); o = np.argsort(t)
    t = t[o]
    ar_int = np.array(ar_int_l)[o]; b_int = np.array(b_int_l)[o]
    Pbaro = np.array(Pbaro_l)[o]; PRey = np.array(PRey_l)[o]
    Pshear = np.array(Pshear_l)[o]; Pb = np.array(Pb_l)[o]
    dadt = np.gradient(ar_int, t); dbdt = np.gradient(b_int, t)
    Psum = Pbaro + PRey + Pshear
    resid_a = dadt - Psum
    resid_b = dbdt - Pb

    print("\n  integral a_r budget dominance (mean |term| over t):")
    print(f"    P_baro={np.mean(np.abs(Pbaro)):.3e}  P_Rey={np.mean(np.abs(PRey)):.3e}  "
          f"P_shear={np.mean(np.abs(Pshear)):.3e}  |d<rho a_r>/dt|={np.mean(np.abs(dadt)):.3e}  "
          f"|resid|={np.mean(np.abs(resid_a)):.3e}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    a = ax[0]
    a.plot(t, Pbaro, lw=2, color="tab:blue", label=r"$P_{baro}=\langle b\,\partial_i p\rangle$")
    a.plot(t, PRey, lw=2, color="tab:orange", label=r"$P_{Rey}=-\langle (R_{ij}/\rho)\partial_j\rho\rangle$")
    a.plot(t, Pshear, lw=2, color="tab:green", label=r"$P_{shear}$")
    a.plot(t, dadt, lw=1.6, ls="--", color="k", label=r"$d\langle\rho a_r\rangle/dt$")
    a.axhline(0, color="gray", lw=0.6)
    a.set_xlabel("t"); a.set_ylabel("a_r budget (radial, vol-int)")
    a.set_title("turbulent mass-flux budget (LES)")
    a.legend(frameon=False, fontsize=8); a.grid(True, ls=":", alpha=0.4)
    a = ax[1]
    a.plot(t, Pb, lw=2, color="tab:purple", label=r"$P_b=-2\langle a_i\partial_i\rho(1+b)\rangle$")
    a.plot(t, dbdt, lw=1.6, ls="--", color="k", label=r"$d\langle\rho b\rangle/dt$")
    a.axhline(0, color="gray", lw=0.6)
    a.set_xlabel("t"); a.set_ylabel("b budget (vol-int)")
    a.set_title("covariance budget (LES)")
    a.legend(frameon=False, fontsize=8); a.grid(True, ls=":", alpha=0.4)
    fig.suptitle("BHR a_r / b budgets measured from the LES ensemble", y=1.02)
    fig.tight_layout()
    out = ROOT / "figs" / "bhr_ab_budget.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out}")
    np.savez(ROOT / "bhr_ab_budget.npz", t=t, ar_int=ar_int, b_int=b_int,
             P_baro=Pbaro, P_Rey=PRey, P_shear=Pshear, P_b=Pb,
             dadt=dadt, dbdt=dbdt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
