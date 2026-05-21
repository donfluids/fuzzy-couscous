#!/usr/bin/env python3
"""Resolution sensor: where is the blast LES resolved vs under-resolved?

Maps the local Kolmogorov scale relative to the grid, eta/dx, across the
chamber. The point is to localise resolution -- the run is known to be ~6x
under-resolved on average (eta << dx), but *where* is it worst?

Honest eps. eta = (nu^3 / eps)^(1/4) is *defined* through the dissipation rate
eps, so an eta/dx map cannot avoid eps entirely. What it CAN avoid is the
unreliable eps: in this LES the resolved viscous/hyper gradient budget
(eps_visc, eps_hyper) under-counts because the SGS sink is numerical. So we read
eps off the *resolved cascade* instead, via a second-order velocity structure
function at a few-cell separation:

    r    = n_cells * dx                       (default n_cells = 2)
    D2   = < |u(x+r) - u(x)|^2 >   over the 6 face neighbours, all 3 components
    eps  = (D2 / C2)^(3/2) / r                (Kolmogorov 2nd-order law, C2~2.0)
    nu   = mu / rho                           (mu = 5e-4 const; nu varies via rho)
    eta  = (nu^3 / eps)^(1/4)
    -> output eta/dx

Validity mask. The cascade estimate is only meaningful where a local inertial
range exists. We compute D2 at r AND 2r and the local slope

    zeta = log2( D2(2r) / D2(r) ).

Inertial range -> zeta ~ 2/3 (eps trustworthy). zeta -> 2 means the field is
smooth/laminar there (or dissipation-range), so the cascade eps -- and hence
eta/dx -- is NOT interpretable; those cells are masked out of the statistics.

Resolution bands (configurable): eta/dx >= 0.5 DNS-resolved; 0.2-0.5 marginal;
< 0.2 under-resolved (LES). We expect the bulk well under 0.2.

Outputs to figs/ and resolution_sensor.npz:
  - mid-plane slices of eta/dx, zeta, and density (context) at a target time
  - ensemble (5-seed) radial profiles eta/dx(r), zeta(r)
  - histogram of eta/dx over valid cells
  - time history of median eta/dx and the band fractions (seed 1)

Reuses the snapshot IO of scripts/bhr_budget.py and the radial-binning idiom of
scripts/dns_radial_average.py. Deliberately does not touch eps_visc/eps_hyper.
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
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
DATA = ROOT / "data"

# Physics / grid (matches blast_128_budget_*.toml).
MU = 5.0e-4          # constant physical viscosity
GAMMA = 1.4
N = 128
L = 1.0
X0 = -0.5
DX = L / N

SEEDS = [1, 2, 3, 4, 5]
RUN_DIR = lambda s: RUNS / f"out_blast_128_budget_seed{s}"
RUN_NAME = lambda s: f"blast_128_budget_seed{s}"

# Sensor defaults.
N_CELLS = 2          # structure-function separation in grid cells (r = n_cells*dx)
C2 = 2.0             # Kolmogorov 2nd-order structure-function constant
TINY = 1e-30

# Resolution bands.
BAND_RESOLVED = 0.5  # eta/dx >= this -> DNS-resolved
BAND_MARGINAL = 0.2  # eta/dx in [0.2,0.5) -> marginal; below -> under-resolved

# Validity: |zeta - 2/3| <= ZETA_TOL counts as "inertial range present".
ZETA_TARGET = 2.0 / 3.0
ZETA_TOL = 0.5       # generous; flags clearly-laminar (zeta near 2) cells out


# ----------------------------------------------------------------------------
# IO  (snapshot listing + loaders; mirrors scripts/bhr_budget.py)
# ----------------------------------------------------------------------------
def list_snapshots(run_dir, run_name):
    """Sorted (step, time, path) for each field snapshot. Excludes the
    checkpoint (.ckpt.h5) and the spectra file (_spectra.h5), which match the
    same glob but carry no 3D primitive fields."""
    pat = re.compile(rf"{re.escape(run_name)}_(\d+)\.h5$")
    out = []
    for p in sorted(run_dir.glob(f"{run_name}_*.h5")):
        if p.name.endswith(".ckpt.h5") or p.name.endswith("_spectra.h5"):
            continue
        m = pat.search(p.name)
        if not m:
            continue
        with h5py.File(p, "r") as f:
            t = float(f["time"][0]) if "time" in f else float("nan")
        out.append((int(m.group(1)), t, p))
    return out


def load_state(p):
    """Return (t, rho, u, v, w, p). Handles both primitive snapshots
    (density, velocity_*, pressure) and conserved checkpoints (rho, rho_u...)."""
    with h5py.File(p, "r") as f:
        keys = set(f.keys())
        if "density" in keys:                       # primitive snapshot
            rho = np.asarray(f["density"], np.float64)
            u = np.asarray(f["velocity_x"], np.float64)
            v = np.asarray(f["velocity_y"], np.float64)
            w = np.asarray(f["velocity_z"], np.float64)
            pr = np.asarray(f["pressure"], np.float64)
            t = float(f["time"][0]) if "time" in keys else float("nan")
        else:                                       # conserved checkpoint
            rho = np.asarray(f["rho"], np.float64)
            u = np.asarray(f["rho_u"], np.float64) / rho
            v = np.asarray(f["rho_v"], np.float64) / rho
            w = np.asarray(f["rho_w"], np.float64) / rho
            pr = np.zeros_like(rho)                  # not needed by the sensor
            t = float(f["time"][0]) if "time" in keys else float("nan")
    return t, rho, u, v, w, pr


# ----------------------------------------------------------------------------
# Sensor core
# ----------------------------------------------------------------------------
def struct_fn_D2(u, v, w, r):
    """Second-order velocity structure function at separation r cells, averaged
    over the 6 face neighbours and all 3 components. np.roll wraps at the
    domain edge; the boundary layer is masked out by `valid_mask` below."""
    D2 = np.zeros_like(u)
    for ax in (0, 1, 2):
        for s in (r, -r):
            du = np.roll(u, -s, axis=ax) - u
            dv = np.roll(v, -s, axis=ax) - v
            dw = np.roll(w, -s, axis=ax) - w
            D2 += du * du + dv * dv + dw * dw
    return D2 / 6.0


def valid_mask(shape, n_cells):
    """Interior cells whose r and 2r face neighbours are real (not wrapped).
    Excludes a 2*n_cells-thick boundary layer (slip walls are not periodic)."""
    m = np.zeros(shape, dtype=bool)
    b = 2 * n_cells
    m[b:-b, b:-b, b:-b] = True
    return m


def compute_sensor(rho, u, v, w, n_cells=N_CELLS, C2=C2, dx=DX, mu=MU):
    """Returns dict with fields eta_dx, zeta, eps, nu, and the validity mask.
    eta/dx and zeta are NaN where invalid (boundary layer or no signal)."""
    r = n_cells
    rsep = r * dx
    D2_r = struct_fn_D2(u, v, w, r)
    D2_2r = struct_fn_D2(u, v, w, 2 * r)

    eps = np.power(np.maximum(D2_r / C2, 0.0), 1.5) / rsep
    nu = mu / rho
    eta = np.power(np.power(nu, 3) / np.maximum(eps, TINY), 0.25)
    eta_dx = eta / dx

    with np.errstate(divide="ignore", invalid="ignore"):
        zeta = np.log2(np.maximum(D2_2r, TINY) / np.maximum(D2_r, TINY))

    interior = valid_mask(rho.shape, n_cells)
    has_signal = D2_r > TINY
    inertial = np.abs(zeta - ZETA_TARGET) <= ZETA_TOL
    valid = interior & has_signal            # for eta/dx stats
    valid_inertial = valid & inertial        # stricter: eps trustworthy

    eta_dx_m = np.where(valid, eta_dx, np.nan)
    zeta_m = np.where(interior & has_signal, zeta, np.nan)
    return dict(eta_dx=eta_dx_m, zeta=zeta_m, eps=eps, nu=nu,
                valid=valid, valid_inertial=valid_inertial)


def band_fractions(eta_dx, valid):
    """Volume fractions in the three resolution bands over valid cells."""
    vals = eta_dx[valid]
    n = vals.size
    if n == 0:
        return dict(resolved=np.nan, marginal=np.nan, under=np.nan,
                    median=np.nan, p10=np.nan, p90=np.nan, n=0)
    resolved = np.mean(vals >= BAND_RESOLVED)
    marginal = np.mean((vals >= BAND_MARGINAL) & (vals < BAND_RESOLVED))
    under = np.mean(vals < BAND_MARGINAL)
    return dict(resolved=float(resolved), marginal=float(marginal),
                under=float(under), median=float(np.median(vals)),
                p10=float(np.percentile(vals, 10)),
                p90=float(np.percentile(vals, 90)), n=int(n))


# ----------------------------------------------------------------------------
# Radial profiles (ensemble + shell averaged; mirrors dns_radial_average.py)
# ----------------------------------------------------------------------------
def cell_radius():
    xc = X0 + (np.arange(N) + 0.5) * DX
    X, Y, Z = np.meshgrid(xc, xc, xc, indexing="ij")
    return np.sqrt(X * X + Y * Y + Z * Z)


def radial_profile(field, valid, r3, nbins=48, r_max=0.5):
    """Shell-average `field` over valid cells. Returns (rmid, mean, count)."""
    bins = np.linspace(0.0, r_max, nbins + 1)
    rmid = 0.5 * (bins[:-1] + bins[1:])
    m = valid & np.isfinite(field) & (r3 <= r_max)
    idx = np.digitize(r3[m], bins) - 1
    ok = (idx >= 0) & (idx < nbins)
    idx = idx[ok]
    vals = field[m][ok]
    cnt = np.bincount(idx, minlength=nbins).astype(float)
    ssum = np.bincount(idx, weights=vals, minlength=nbins)
    mean = np.where(cnt > 0, ssum / np.maximum(cnt, 1), np.nan)
    return rmid, mean, cnt


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def nearest_snapshot(snaps, t_target):
    return min(snaps, key=lambda s: abs(s[1] - t_target))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t-map", type=float, default=0.12,
                    help="target time for slices/radial/histogram (default 0.12)")
    ap.add_argument("--n-cells", type=int, default=N_CELLS,
                    help="structure-function separation in cells (default 2)")
    ap.add_argument("--C2", type=float, default=C2,
                    help="Kolmogorov 2nd-order constant (default 2.0)")
    ap.add_argument("--seed", type=int, default=1,
                    help="seed used for slices and the time history (default 1)")
    args = ap.parse_args()

    figs = ROOT / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    r3 = cell_radius()

    # --- snapshot for spatial maps (chosen seed, nearest t_map) ---
    snaps = list_snapshots(RUN_DIR(args.seed), RUN_NAME(args.seed))
    if not snaps:
        print(f"No snapshots in {RUN_DIR(args.seed)}", file=sys.stderr)
        return 1
    step, t, path = nearest_snapshot(snaps, args.t_map)
    _, rho, u, v, w, _ = load_state(path)
    S = compute_sensor(rho, u, v, w, args.n_cells, args.C2)
    bf = band_fractions(S["eta_dx"], S["valid"])
    print(f"[map] seed{args.seed} step={step} t={t:.4f}  "
          f"median eta/dx={bf['median']:.3f} (p10={bf['p10']:.3f}, "
          f"p90={bf['p90']:.3f})")
    print(f"      bands: resolved={bf['resolved']*100:.1f}%  "
          f"marginal={bf['marginal']*100:.1f}%  "
          f"under={bf['under']*100:.1f}%  (n_valid={bf['n']})")
    frac_inert = float(np.mean(S["valid_inertial"][S["valid"]])) if bf["n"] else float("nan")
    print(f"      inertial-range valid fraction (|zeta-2/3|<={ZETA_TOL}): "
          f"{frac_inert*100:.1f}%")

    # --- ensemble radial profiles at t_map (all seeds) ---
    eta_prof_acc, zeta_prof_acc, cnt_acc = None, None, None
    rmid = None
    used = 0
    for s in SEEDS:
        sn = list_snapshots(RUN_DIR(s), RUN_NAME(s))
        if not sn:
            continue
        st, ts, pa = nearest_snapshot(sn, args.t_map)
        _, rh, uu, vv, ww, _ = load_state(pa)
        Ss = compute_sensor(rh, uu, vv, ww, args.n_cells, args.C2)
        rmid, em, ec = radial_profile(Ss["eta_dx"], Ss["valid"], r3)
        _, zm, _ = radial_profile(Ss["zeta"], Ss["valid"], r3)
        if eta_prof_acc is None:
            eta_prof_acc = np.zeros_like(em)
            zeta_prof_acc = np.zeros_like(zm)
            cnt_acc = np.zeros_like(ec)
        # count-weighted accumulation across seeds
        eta_prof_acc += np.nan_to_num(em) * ec
        zeta_prof_acc += np.nan_to_num(zm) * ec
        cnt_acc += ec
        used += 1
    eta_prof = np.where(cnt_acc > 0, eta_prof_acc / np.maximum(cnt_acc, 1), np.nan)
    zeta_prof = np.where(cnt_acc > 0, zeta_prof_acc / np.maximum(cnt_acc, 1), np.nan)
    print(f"[radial] ensemble of {used} seeds at t~{args.t_map}")

    # --- time history (chosen seed, all snapshots) ---
    th_t, th_med, th_res, th_marg, th_under = [], [], [], [], []
    for st, ts, pa in snaps:
        if ts <= 0:
            continue
        _, rh, uu, vv, ww, _ = load_state(pa)
        Ss = compute_sensor(rh, uu, vv, ww, args.n_cells, args.C2)
        b = band_fractions(Ss["eta_dx"], Ss["valid"])
        th_t.append(ts); th_med.append(b["median"])
        th_res.append(b["resolved"]); th_marg.append(b["marginal"])
        th_under.append(b["under"])
    th_t = np.array(th_t)
    order = np.argsort(th_t)
    th_t = th_t[order]
    th_med = np.array(th_med)[order]
    th_res = np.array(th_res)[order]
    th_marg = np.array(th_marg)[order]
    th_under = np.array(th_under)[order]
    print(f"[time] seed{args.seed}: {len(th_t)} snapshots, "
          f"t in [{th_t.min():.3f}, {th_t.max():.3f}]")

    # ------------------------------------------------------------------ plots
    mid = N // 2

    # Figure 1: slices (eta/dx, zeta, density context).
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ext = [X0, X0 + L, X0, X0 + L]
    eta_slice = S["eta_dx"][:, :, mid].T
    im0 = ax[0].imshow(eta_slice, origin="lower", extent=ext,
                       norm=LogNorm(vmin=0.03, vmax=1.0), cmap="viridis")
    ax[0].set_title(r"$\eta/\Delta x$  (mid-plane)")
    plt.colorbar(im0, ax=ax[0], fraction=0.046)
    # mark the resolved threshold contour
    ax[0].contour(np.linspace(*ext[:2], N), np.linspace(*ext[2:], N),
                  np.nan_to_num(eta_slice), levels=[BAND_MARGINAL, BAND_RESOLVED],
                  colors=["white", "red"], linewidths=0.8)

    zeta_slice = S["zeta"][:, :, mid].T
    im1 = ax[1].imshow(zeta_slice, origin="lower", extent=ext,
                       vmin=0.0, vmax=2.0, cmap="coolwarm")
    ax[1].set_title(r"slope $\zeta=\log_2 D_2(2r)/D_2(r)$"
                    "\n(2/3 inertial, 2 laminar)")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)

    im2 = ax[2].imshow(rho[:, :, mid].T, origin="lower", extent=ext, cmap="magma")
    ax[2].set_title(r"$\rho$  (context)")
    plt.colorbar(im2, ax=ax[2], fraction=0.046)
    for a in ax:
        a.set_xlabel("x"); a.set_ylabel("y")
    fig.suptitle(f"Resolution sensor  seed{args.seed}  t={t:.3f}  "
                 f"(r={args.n_cells} dx, C2={args.C2})", y=1.02)
    fig.tight_layout()
    f1 = figs / "resolution_slices.png"
    fig.savefig(f1, dpi=140, bbox_inches="tight")
    print(f"wrote {f1}")

    # Figure 2: radial profile + histogram.
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(rmid, eta_prof, lw=2, color="tab:blue", label=r"$\eta/\Delta x$")
    ax[0].axhline(BAND_RESOLVED, color="red", ls="--", lw=1,
                  label=f"resolved ({BAND_RESOLVED})")
    ax[0].axhline(BAND_MARGINAL, color="gray", ls=":", lw=1,
                  label=f"marginal ({BAND_MARGINAL})")
    ax[0].set_xlabel("r"); ax[0].set_ylabel(r"$\eta/\Delta x$")
    ax[0].set_yscale("log")
    ax[0].set_title(rf"ensemble radial profile, t$\approx${args.t_map}")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].grid(True, which="both", ls=":", alpha=0.4)
    axb = ax[0].twinx()
    axb.plot(rmid, zeta_prof, lw=1.3, color="tab:green", alpha=0.7)
    axb.axhline(ZETA_TARGET, color="tab:green", ls=":", lw=0.8)
    axb.set_ylabel(r"$\zeta$ (green)", color="tab:green")
    axb.set_ylim(0, 2.1)

    vals = S["eta_dx"][S["valid"]]
    ax[1].hist(vals, bins=np.logspace(-2, 0.3, 60), color="tab:blue", alpha=0.8)
    ax[1].axvline(BAND_RESOLVED, color="red", ls="--", lw=1)
    ax[1].axvline(BAND_MARGINAL, color="gray", ls=":", lw=1)
    ax[1].axvline(bf["median"], color="k", lw=1.5,
                  label=f"median={bf['median']:.3f}")
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$\eta/\Delta x$"); ax[1].set_ylabel("cells")
    ax[1].set_title(f"distribution (seed{args.seed}, t={t:.3f})")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    f2 = figs / "resolution_radial_hist.png"
    fig.savefig(f2, dpi=140, bbox_inches="tight")
    print(f"wrote {f2}")

    # Figure 3: time history.
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(th_t, th_med, lw=2, color="tab:blue")
    ax[0].axhline(BAND_RESOLVED, color="red", ls="--", lw=1)
    ax[0].axhline(BAND_MARGINAL, color="gray", ls=":", lw=1)
    ax[0].set_xlabel("t"); ax[0].set_ylabel(r"median $\eta/\Delta x$")
    ax[0].set_yscale("log")
    ax[0].set_title(f"median resolution vs time (seed{args.seed})")
    ax[0].grid(True, which="both", ls=":", alpha=0.4)

    ax[1].stackplot(th_t, th_under, th_marg, th_res,
                    labels=["under-resolved", "marginal", "resolved"],
                    colors=["tab:red", "tab:orange", "tab:green"], alpha=0.85)
    ax[1].set_xlabel("t"); ax[1].set_ylabel("volume fraction")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("resolution-band fractions vs time")
    ax[1].legend(loc="center right", frameon=False, fontsize=8)
    fig.tight_layout()
    f3 = figs / "resolution_timehistory.png"
    fig.savefig(f3, dpi=140, bbox_inches="tight")
    print(f"wrote {f3}")

    # ------------------------------------------------------------------ npz
    out = DATA / "resolution_sensor.npz"
    np.savez(out,
             t_map=t, step_map=step, seed=args.seed,
             n_cells=args.n_cells, C2=args.C2,
             eta_dx_slice=S["eta_dx"][:, :, mid],
             zeta_slice=S["zeta"][:, :, mid],
             rho_slice=rho[:, :, mid],
             rmid=rmid, eta_dx_radial=eta_prof, zeta_radial=zeta_prof,
             th_t=th_t, th_median=th_med, th_resolved=th_res,
             th_marginal=th_marg, th_under=th_under,
             bands=np.array([bf["resolved"], bf["marginal"], bf["under"]]))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
