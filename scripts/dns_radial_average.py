#!/usr/bin/env python3
"""Spherically + ensemble averaged radial profiles from the DNS budget ensemble.

For each target time, loads all 5 budget-seed snapshots near that time and
bins every cell by radius r = |x| (box centered at origin). Produces, per
radial shell, the ensemble+shell mean fields and the turbulence quantities
that the 1D BHR RANS predicts:

  rho_bar(r), u_r_bar(r), p_bar(r)
  k(r)   = 1/2 [ <(u_r - u_r_bar)^2> + <u_tang^2> ]   (radial + tangential TKE)
  a_r(r) = <rho' u_r'> / rho_bar                        (turbulent mass flux)
  b(r)   = rho_bar * <1/rho> - 1                        (density-spec-vol cov)

Single-pass moment accumulation (histogram of sums) across all 5 seeds.
Output: dns_radial_profiles.npz (list of per-time dicts).
"""

import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [1, 2, 3, 4, 5]
RUN_DIR = lambda s: ROOT / f"out_blast_128_budget_seed{s}"
RUN_NAME = lambda s: f"blast_128_budget_seed{s}"

N = 128
L = 1.0
X0 = -0.5
DX = L / N
R_MAX = 0.5            # compare within the inscribed sphere (wall radius)
NBINS = 64

TARGET_TIMES = [0.02, 0.05, 0.10, 0.25]


def cell_radius():
    xc = X0 + (np.arange(N) + 0.5) * DX
    X, Y, Z = np.meshgrid(xc, xc, xc, indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z)
    # unit radial vector components
    rinv = 1.0 / np.maximum(r, 1e-30)
    return r, X * rinv, Y * rinv, Z * rinv


def snapshot_table(seed):
    import re
    d = RUN_DIR(seed)
    name = RUN_NAME(seed)
    pat = re.compile(rf"{re.escape(name)}_(\d+)\.h5$")
    out = {}
    for p in sorted(d.glob(f"{name}_*.h5")):
        if p.name.endswith(".ckpt.h5"):
            continue
        m = pat.search(p.name)
        if not m:
            continue
        with h5py.File(p, "r") as f:
            t = float(f["time"][0])
        out[int(m.group(1))] = (t, p)
    return out


def load_prim(p):
    with h5py.File(p, "r") as f:
        rho = np.array(f["density"], dtype=np.float64)
        ux = np.array(f["velocity_x"], dtype=np.float64)
        uy = np.array(f["velocity_y"], dtype=np.float64)
        uz = np.array(f["velocity_z"], dtype=np.float64)
        pr = np.array(f["pressure"], dtype=np.float64)
    return rho, ux, uy, uz, pr


def main():
    r, ehx, ehy, ehz = cell_radius()
    bins = np.linspace(0.0, R_MAX, NBINS + 1)
    rmid = 0.5 * (bins[:-1] + bins[1:])
    rflat = r.ravel()
    inside = rflat <= R_MAX
    ehx = ehx.ravel(); ehy = ehy.ravel(); ehz = ehz.ravel()

    tables = {s: snapshot_table(s) for s in SEEDS}
    common_steps = set.intersection(*[set(t.keys()) for t in tables.values()])

    profiles = []
    for t_target in TARGET_TIMES:
        # pick the step whose seed-averaged time is closest to t_target
        best_step, best_dt = None, 1e9
        for step in common_steps:
            tavg = np.mean([tables[s][step][0] for s in SEEDS])
            if abs(tavg - t_target) < best_dt:
                best_dt, best_step = abs(tavg - t_target), step
        step = best_step
        tavg = np.mean([tables[s][step][0] for s in SEEDS])

        # Accumulate moment sums over all seeds.
        S_cnt = np.zeros(NBINS)
        S_rho = np.zeros(NBINS); S_invrho = np.zeros(NBINS)
        S_ur = np.zeros(NBINS); S_ur2 = np.zeros(NBINS)
        S_ut2 = np.zeros(NBINS)
        S_p = np.zeros(NBINS)
        S_rhour = np.zeros(NBINS)

        for s in SEEDS:
            rho, ux, uy, uz, pr = load_prim(tables[s][step][1])
            rho = rho.ravel(); ux = ux.ravel(); uy = uy.ravel()
            uz = uz.ravel(); pr = pr.ravel()
            u_r = ux * ehx + uy * ehy + uz * ehz          # radial velocity
            u2 = ux * ux + uy * uy + uz * uz
            u_tang2 = np.maximum(u2 - u_r * u_r, 0.0)

            m = inside
            idx = np.digitize(rflat[m], bins) - 1
            valid = (idx >= 0) & (idx < NBINS)
            idx = idx[valid]
            def acc(arr):
                return np.bincount(idx, weights=arr[m][valid], minlength=NBINS)
            S_cnt += np.bincount(idx, minlength=NBINS)
            S_rho += acc(rho); S_invrho += acc(1.0 / rho)
            S_ur += acc(u_r); S_ur2 += acc(u_r * u_r)
            S_ut2 += acc(u_tang2)
            S_p += acc(pr)
            S_rhour += acc(rho * u_r)

        cnt = np.maximum(S_cnt, 1)
        rho_bar = S_rho / cnt
        invrho_bar = S_invrho / cnt
        ur_bar = S_ur / cnt
        p_bar = S_p / cnt
        var_ur = np.maximum(S_ur2 / cnt - ur_bar ** 2, 0.0)
        ut2_bar = S_ut2 / cnt
        k = 0.5 * (var_ur + ut2_bar)
        cov_rho_ur = S_rhour / cnt - rho_bar * ur_bar
        a_r = cov_rho_ur / np.maximum(rho_bar, 1e-30)
        b = rho_bar * invrho_bar - 1.0

        profiles.append(dict(t=tavg, step=step, r=rmid,
                             rho=rho_bar, u_r=ur_bar, p=p_bar,
                             k=k, a_r=a_r, b=b, count=S_cnt))
        print(f"t_target={t_target:.3f} -> step={step} t={tavg:.4f} "
              f"k_peak={k.max():.4e} a_peak={np.abs(a_r).max():.4e} "
              f"b_peak={b.max():.4e}")

    np.savez(ROOT / "dns_radial_profiles.npz",
             profiles=np.array(profiles, dtype=object))
    print(f"Wrote {ROOT / 'dns_radial_profiles.npz'} ({len(profiles)} times)")


if __name__ == "__main__":
    main()
