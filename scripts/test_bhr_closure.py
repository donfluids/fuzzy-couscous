#!/usr/bin/env python3
"""Test whether BHR-family closure variables (turbulent mass flux a_i,
density-specific-volume covariance b) are significant in the 128^3 slip-wall
blast ensemble.

Reads the 5 final checkpoints (baseline + 4 ensemble seeds), forms pointwise
ensemble averages, computes:

  ensemble averages: <rho>, <u_i>, <p>, <v = 1/rho>
  fluctuations:      rho' = rho - <rho>, u_i' = u_i - <u_i>, v' = v - <v>
  BHR variables:
      a_i  = <rho' u_i'> / <rho>
      b    = -<rho' v'>  =  <rho>*<v> - 1

Reports global magnitudes and writes a 2D slice plot through z=0.

If |a_i| and b are negligible compared to natural scales (b<<1, |a|<<u_rms),
standard k-eps suffices. If they are O(1)*u_rms, BHR tracking is required.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
GAMMA = 1.4

RUNS = [
    "out_blast_128_slip_hyper6_fd",
    "out_blast_128_slip_hyper6_fd_seed1",
    "out_blast_128_slip_hyper6_fd_seed2",
    "out_blast_128_slip_hyper6_fd_seed3",
    "out_blast_128_slip_hyper6_fd_seed4",
]


def load_ckpt(out_dir):
    name = out_dir.replace("out_", "")
    ckpt = ROOT / out_dir / f"{name}.ckpt.h5"
    with h5py.File(ckpt, "r") as f:
        rho  = np.array(f["rho"],   dtype=np.float64)
        rhou = np.array(f["rho_u"], dtype=np.float64)
        rhov = np.array(f["rho_v"], dtype=np.float64)
        rhow = np.array(f["rho_w"], dtype=np.float64)
        rhoE = np.array(f["rho_E"], dtype=np.float64)
        t    = float(f["time"][0])
    u = rhou / rho
    v = rhov / rho
    w = rhow / rho
    e_int = (rhoE - 0.5*(rhou*u + rhov*v + rhow*w)) / rho
    p = (GAMMA - 1.0) * rho * e_int
    return t, rho, u, v, w, p


def main():
    fields = []
    for r in RUNS:
        try:
            fields.append(load_ckpt(r))
        except (OSError, KeyError) as e:
            print(f"skip {r}: {e}", file=sys.stderr)
    N = len(fields)
    if N < 2:
        print("Need at least 2 ensemble members.", file=sys.stderr)
        return 1

    times = np.array([f[0] for f in fields])
    rhos  = np.stack([f[1] for f in fields])
    us    = np.stack([f[2] for f in fields])
    vs    = np.stack([f[3] for f in fields])
    ws    = np.stack([f[4] for f in fields])
    ps    = np.stack([f[5] for f in fields])

    print(f"Ensemble N={N} members; checkpoint times {times}")

    # Pointwise ensemble averages.
    rho_m = rhos.mean(axis=0)
    u_m   = us.mean(axis=0)
    v_m   = vs.mean(axis=0)
    w_m   = ws.mean(axis=0)
    p_m   = ps.mean(axis=0)

    # Fluctuations.
    rho_p = rhos - rho_m
    u_p   = us - u_m
    v_p   = vs - v_m
    w_p   = ws - w_m

    # Turbulent mass flux a_i = <rho' u'_i> / <rho>.
    a_x = (rho_p * u_p).mean(axis=0) / rho_m
    a_y = (rho_p * v_p).mean(axis=0) / rho_m
    a_z = (rho_p * w_p).mean(axis=0) / rho_m
    a_mag = np.sqrt(a_x**2 + a_y**2 + a_z**2)

    # Density-spec-vol covariance b = -<rho' v'> = <rho>*<v> - 1, v=1/rho.
    spec_vols = 1.0 / rhos
    v_m_inv = spec_vols.mean(axis=0)
    b_field = rho_m * v_m_inv - 1.0

    # Solenoidal-ish K from velocity fluctuations.
    k_field = 0.5 * (u_p**2 + v_p**2 + w_p**2).mean(axis=0)

    # Reynolds stress (kinetic-form). Diagonal trace = 2 k.
    R_xx = (u_p * u_p).mean(axis=0)
    R_yy = (v_p * v_p).mean(axis=0)
    R_zz = (w_p * w_p).mean(axis=0)

    # Pressure-gradient (for variable-density production term).
    h = 1.0 / 128.0
    dpx = np.gradient(p_m, h, axis=0)
    dpy = np.gradient(p_m, h, axis=1)
    dpz = np.gradient(p_m, h, axis=2)

    # Variable-density production:  P_VD = - a_i * dp_m/dx_i  (BHR form,
    # mass-flux times mean pressure gradient).
    P_VD = -(a_x * dpx + a_y * dpy + a_z * dpz)

    # Diagnostic magnitudes.
    def rms(x): return float(np.sqrt(np.mean(x * x)))
    def vol(x): return float(np.mean(x))

    u_rms = float(np.sqrt(np.mean(u_p**2 + v_p**2 + w_p**2)))
    rho_rms = float(np.sqrt(np.mean(rho_p**2)))
    p_rms = float(np.sqrt(np.mean((ps - p_m)**2)))

    print()
    print("=== Global magnitudes ===")
    print(f"  rho_m mean       = {vol(rho_m):.4f}")
    print(f"  rho' RMS         = {rho_rms:.4f}   (rho'/<rho> = {rho_rms/vol(rho_m):.3f})")
    print(f"  |u'| RMS (3D)    = {u_rms:.4f}")
    print(f"  k (turbulent KE) = {vol(k_field):.4f}")
    print()
    print("=== BHR variables ===")
    print(f"  |a| RMS          = {rms(a_mag):.4e}  ({rms(a_mag)/u_rms:.3f} u_rms)")
    print(f"  a_x RMS          = {rms(a_x):.4e}")
    print(f"  a_y RMS          = {rms(a_y):.4e}")
    print(f"  a_z RMS          = {rms(a_z):.4e}")
    print(f"  b mean           = {vol(b_field):.4e}  (=0 if rho is uniform)")
    print(f"  b RMS            = {rms(b_field):.4e}")
    print()
    print("=== Variable-density production term ===")
    print(f"  <|dp/dx|>             = {rms(np.stack([dpx,dpy,dpz])):.4e}")
    print(f"  <P_VD> = <-a.grad p>  = {vol(P_VD):.4e}")
    print(f"  P_VD RMS              = {rms(P_VD):.4e}")
    print()
    print("=== Verdict heuristics ===")
    threshold_a = 0.05 * u_rms
    threshold_b = 1e-3
    a_signif = rms(a_mag) > threshold_a
    b_signif = rms(b_field) > threshold_b
    print(f"  |a| > 5% u_rms? {a_signif}    (rms |a| = {rms(a_mag):.3e},"
          f" 0.05 u_rms = {threshold_a:.3e})")
    print(f"  b  > 1e-3?      {b_signif}    (rms b   = {rms(b_field):.3e})")
    if a_signif and b_signif:
        print("  -> BHR-style closure WARRANTED: both a_i and b are non-negligible.")
    elif a_signif or b_signif:
        print("  -> Marginal: one BHR variable is significant. Worth carrying.")
    else:
        print("  -> Standard k-eps could plausibly cover this regime "
              "(but K_sol/K_total partition still matters).")

    # Plots: slices through z = N/2.
    iz = rho_m.shape[2] // 2
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    def show(ax, fld, title, cmap="viridis", **kw):
        im = ax.imshow(fld[:, :, iz].T, origin="lower", cmap=cmap, **kw)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8)

    show(axes[0,0], rho_m,
         r"$\langle\rho\rangle$ (ensemble mean)")
    show(axes[0,1], np.sqrt((rho_p**2).mean(axis=0)),
         r"$\sigma_\rho$ (RMS of $\rho'$)",
         cmap="magma")
    show(axes[0,2], k_field,
         r"$k$ = $\frac{1}{2}\langle u'_i u'_i\rangle$",
         cmap="plasma")

    a_lim = max(np.percentile(np.abs(a_mag), 99), 1e-12)
    show(axes[1,0], a_mag,
         r"$|a| = |\langle\rho' u'\rangle|/\langle\rho\rangle$",
         cmap="viridis", vmin=0, vmax=a_lim)
    show(axes[1,1], b_field,
         r"$b = \langle\rho\rangle\langle 1/\rho\rangle - 1$",
         cmap="magma", vmin=0)
    show(axes[1,2], P_VD,
         r"$P_{\rm VD} = -a_i\,\partial_i \bar p$ (BHR VD production)",
         cmap="seismic",
         vmin=-rms(P_VD), vmax=rms(P_VD))

    fig.suptitle(
        rf"BHR-closure test on 128$^3$ blast ensemble ($N={N}$, $t\approx{times.mean():.3f}$): "
        r"slice at $z=0$",
        y=1.02, fontsize=12)
    out = ROOT / "figs" / "bhr_closure_test.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
