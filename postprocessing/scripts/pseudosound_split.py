#!/usr/bin/env python3
"""Pseudo-sound vs. true-sound split of the compressible (dilatational) field.

The Helmholtz split (E_dil, K_dil) lumps ALL curl-free motion together, but that
field contains two physically distinct things:

  * **pseudo-sound** -- the pressure/dilatation slaved instantaneously to the
    (mostly vortical) turbulence. This is the genuine *dilatational turbulence*.
  * **true sound** -- freely-propagating + closed-chamber standing acoustic
    waves / blast ringing. This carries compressible energy but does not cascade.

We separate them through the pressure. Taking the divergence of the compressible
momentum equation gives an exact Poisson relation for the pressure,

    laplacian(p) = -d_i d_j (rho u_i u_j)  -  d^2 rho / dt^2 .

The pseudo-sound pressure is the part slaved to the *vortical* turbulence
(Ristorcelli 1997, Sarkar): the dilatational field is driven by the solenoidal
field, so the pseudo-sound Poisson equation is sourced by the SOLENOIDAL
velocity u_s = Helmholtz-solenoidal(u),

    laplacian(p_h) = -d_i d_j (rho u_s,i u_s,j)                  (Poisson, FFT)

and the acoustic pressure is the remainder  p_a = p' - p_h  (it carries the
d^2 rho / dt^2 content without our needing a time derivative). Sourcing with the
*full* velocity would be circular here -- this field is overwhelmingly
dilatational, so the full-velocity "incompressible pressure" just re-absorbs the
acoustics it is meant to remove. In Fourier space the Poisson solve is the
scale-invariant double projection onto k-hat,

    p_h(k) = -(k_i k_j / |k|^2) (rho u_s,i u_s,j)^(k) ,   p_h(0) = 0 ,

so it is independent of box size and reuses the same FFT grid as the other
diagnostics (scripts/intermode_transfer.py: grids/helmholtz/load).

Outputs, time-averaged over a window:
  * pressure variances <p'^2>, <p_h^2> (pseudo-sound), <p_a^2> (acoustic), their
    cross-covariance, and the acoustic pressure-intensity ratio;
  * pressure spectra P_tot(k), P_h(k), P_a(k) and the acoustic fraction vs k;
  * a KE bridge: acoustic specific KE e_ac = <p_a^2>/(2 rho_bar^2 c_bar^2)
    (equipartition closure) vs the Helmholtz K_dil, giving an *estimate* of the
    turbulent-dilatational KE  K_dil - e_ac.

CAVEATS (carry into the manuscript):
  * The acoustic/pseudo-sound split of the *kinetic* energy uses the equipartition
    closure (acoustic KE ~ acoustic PE); the pressure split itself is exact, the
    KE attribution is an estimate.
  * p_h and p_a are not orthogonal, so variances are not strictly additive (the
    cross-covariance is reported).
  * The momentum-flux source neglects viscous/baroclinic Poisson sources; in a
    strong blast with shocklets the low-Mach pseudo-sound picture is approximate.
  * Periodic-FFT basis -- an approximation near slip walls (as in intermode_transfer).
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

# Reuse the shared, N-agnostic FFT primitives.
from intermode_transfer import grids, helmholtz, kbin, nbin, kphys  # noqa: E402


def load_fields(path):
    """Load (t, rho, u, v, w, p), transposing HDF5 [z,y,x] -> [x,y,z]."""
    with h5py.File(path, "r") as f:
        T = lambda a: np.transpose(np.asarray(a, np.float64), (2, 1, 0))
        return (float(f["time"][0]), T(f["density"]),
                T(f["velocity_x"]), T(f["velocity_y"]), T(f["velocity_z"]),
                T(f["pressure"]))


def pseudosound_pressure(rho, vel):
    """Pseudo-sound pressure p_h via laplacian(p_h) = -d_i d_j(rho v_i v_j).

    ``vel`` is the velocity that *sources* the pressure -- pass the SOLENOIDAL
    velocity u_s for the genuine pseudo-sound (pressure slaved to the vortical
    turbulence). Returns (p_h, ph_hat), both zero mean. In Fourier:
        p_h(k) = -(k_i k_j (rho v_i v_j)^) / |k|^2 .
    """
    g = grids(rho.shape[0])
    K2 = g["K2"]
    K = (g["KX"], g["KY"], g["KZ"])
    num = np.zeros_like(K2, dtype=complex)
    for i in range(3):
        for j in range(3):
            Tij = np.fft.fftn(rho * vel[i] * vel[j])
            num += K[i] * K[j] * Tij
    ph_hat = -num / K2          # K2[0,0,0] set to 1 by grids()
    ph_hat[0, 0, 0] = 0.0       # zero-mean pseudo-sound pressure
    p_h = np.fft.ifftn(ph_hat).real
    return p_h, ph_hat


def shell_spectrum(field_hat, N):
    """Shell-binned |field_hat|^2 normalised so sum_k P(k) = <field^2>."""
    KB = kbin(N)
    w = (np.abs(field_hat) ** 2).ravel() / N ** 6
    return np.bincount(KB.ravel(), weights=w, minlength=nbin(N))[:nbin(N)]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="out_blast_128_budget_seed1")
    ap.add_argument("--name", default="blast_128_budget_seed1")
    ap.add_argument("--t0", type=float, default=0.01)
    ap.add_argument("--t1", type=float, default=0.02)
    ap.add_argument("--gamma", type=float, default=1.4, help="ratio of specific heats")
    ap.add_argument("--label", default=None, help="output tag (default: from run)")
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
    label = args.label or args.run.replace("out_", "")
    print(f"{len(sel)} snapshots in [{args.t0},{args.t1}], N={N}, gamma={args.gamma}")

    nb = nbin(N)
    P_tot = np.zeros(nb); P_h = np.zeros(nb); P_a = np.zeros(nb)
    var_p = var_h = var_a = cov = 0.0
    e_ac = K_dil = K_sol = 0.0
    tlo, thi = 1e9, -1e9
    for t, p in sel:
        _, rho, u, v, w, pr = load_fields(p)
        pp = pr - pr.mean()                       # pressure fluctuation
        us, ud = helmholtz(u, v, w)               # source p_h with solenoidal u
        p_h, ph_hat = pseudosound_pressure(rho, us)
        p_a = pp - p_h                            # acoustic remainder

        var_p += np.mean(pp ** 2)
        var_h += np.mean(p_h ** 2)
        var_a += np.mean(p_a ** 2)
        cov += np.mean(p_h * p_a)

        P_tot += shell_spectrum(np.fft.fftn(pp), N)
        P_h += shell_spectrum(ph_hat, N)
        P_a += shell_spectrum(np.fft.fftn(p_a), N)

        # KE bridge (equipartition estimate of acoustic specific KE)
        rho_bar = rho.mean()
        c2 = args.gamma * pr.mean() / rho_bar
        e_ac += np.mean(p_a ** 2) / (2.0 * rho_bar ** 2 * c2)
        K_dil += 0.5 * np.mean(sum(ud[i] ** 2 for i in range(3)))
        K_sol += 0.5 * np.mean(sum(us[i] ** 2 for i in range(3)))
        tlo, thi = min(tlo, t), max(thi, t)
        print(f"  t={t:.4f}  <p_h^2>/<p'^2>={np.mean(p_h**2)/np.mean(pp**2):.3f}  "
              f"K_dil={0.5*np.mean(sum(ud[i]**2 for i in range(3))):.3e}")

    n = len(sel)
    for arr in (P_tot, P_h, P_a):
        arr /= n
    var_p, var_h, var_a, cov = var_p/n, var_h/n, var_a/n, cov/n
    e_ac, K_dil, K_sol = e_ac/n, K_dil/n, K_sol/n
    k = kphys(N)

    print(f"\n  === PRESSURE split (exact: p' = p_h + p_a) ===")
    print(f"    <p'^2>            = {var_p:.4e}")
    print(f"    <p_h^2> pseudo    = {var_h:.4e}   ({var_h/var_p:6.3f} of <p'^2>)")
    print(f"    <p_a^2> acoustic  = {var_a:.4e}   ({var_a/var_p:6.3f} of <p'^2>)")
    print(f"    2<p_h p_a> cross  = {2*cov:.4e}   "
          f"(check: p_h+p_a recovers {(var_h+var_a+2*cov)/var_p:.4f} of <p'^2>)")

    print(f"\n  === KE bridge (equipartition estimate; specific KE) ===")
    print(f"    K_dil (Helmholtz)      = {K_dil:.4e}")
    print(f"    e_ac  acoustic KE est  = {e_ac:.4e}   ({e_ac/max(K_dil,1e-30):6.3f} of K_dil)")
    if e_ac < K_dil:
        print(f"    K_dil - e_ac (turb dil)= {K_dil - e_ac:.4e}   "
              f"({(K_dil-e_ac)/max(K_dil,1e-30):6.3f} of K_dil)")
    else:
        print(f"    turb-dil KE: e_ac exceeds K_dil -> equipartition closure invalid "
              f"in this regime; use the pressure split, not the KE bridge.")
    print(f"    [for reference K_sol   = {K_sol:.4e}]")

    # ----------------------------------------------------------------- plots
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    a = ax[0]
    for arr, lab, col in ((P_tot, "P_tot (p')", "k"),
                          (P_h, "P_h pseudo-sound (turbulent)", "tab:blue"),
                          (P_a, "P_a acoustic (true sound)", "tab:red")):
        m = (k > 0) & (arr > 0)
        a.loglog(k[m], arr[m], lw=2, color=col, label=lab)
    a.set_xlabel("k"); a.set_ylabel(r"pressure spectrum $P(k)$")
    a.set_title("pressure: pseudo-sound vs true sound")
    a.legend(frameon=False, fontsize=8); a.grid(True, which="both", ls=":", alpha=0.4)

    a = ax[1]
    m = (k > 0) & (P_tot > 0)
    a.semilogx(k[m], P_a[m] / P_tot[m], lw=2, color="tab:red", label=r"$P_a/P_{tot}$")
    a.semilogx(k[m], P_h[m] / P_tot[m], lw=2, color="tab:blue", label=r"$P_h/P_{tot}$")
    a.axhline(0.5, color="gray", ls="--", lw=1)
    a.set_ylim(0, 1.05); a.set_xlabel("k")
    a.set_ylabel("fraction of pressure spectrum")
    a.set_title("acoustic vs pseudo-sound fraction by scale")
    a.legend(frameon=False, fontsize=8); a.grid(True, which="both", ls=":", alpha=0.4)

    fig.suptitle(f"Pseudo-sound split  {args.run}  t-avg [{tlo:.3f},{thi:.3f}] "
                 f"({n} snaps)   acoustic={var_a/var_p:.2f} of <p'^2>", y=1.02)
    fig.tight_layout()
    out = ROOT / "figs" / f"pseudosound_split_{label}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out}")

    np.savez(DATA / f"pseudosound_split_{label}.npz", k=k, P_tot=P_tot, P_h=P_h,
             P_a=P_a, var_p=var_p, var_h=var_h, var_a=var_a, cov=cov,
             e_ac=e_ac, K_dil=K_dil, K_sol=K_sol, N=N, gamma=args.gamma,
             window=(tlo, thi))
    return 0


if __name__ == "__main__":
    sys.exit(main())
