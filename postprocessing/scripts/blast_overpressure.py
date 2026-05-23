#!/usr/bin/env python3
"""Peak-overpressure decay of a free-air blast (e.g. the JWL TNT run).

For each snapshot, bins every cell by radius r=|x| (box centered at origin) and
takes the shell-MAXIMUM pressure -- the leading shock overpressure at that
radius. Tracks the shock-front radius R_s(t) (radius of the global pressure peak
outside the products core) and the peak overpressure there. Produces:

  (1) peak overpressure dp = p_peak - p_amb vs scaled distance Z = R_s / r0,
      the canonical blast-decay curve (should fall off as a steep power law and
      flatten toward a weak acoustic wave),
  (2) the shock-front trajectory R_s(t) (deceleration toward the ambient sound
      speed), and
  (3) radial shell-max pressure profiles at several times.

Quantitative Kingery-Bulmash / Sadovskii comparison needs the dimensional charge
mass (TNT-equivalent W) and the L_ref length scale; here we report the
nondimensional decay and trajectory, which already validate that the JWL blast
forms a decelerating shock that decays with distance. Pass p_ref to convert the
overpressure axis to physical units.

Usage:
  python scripts/blast_overpressure.py [run_dir] [run_name] [--r0 R] [--pamb P]
Defaults target runs/out_tnt_freeair_128.
"""

import re
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


def parse_args(argv):
    run_dir = run_dir("out_tnt_freeair_128")
    run_name = "tnt_freeair_128"
    r0 = 0.06
    p_amb = 1.013e5 / 21.0e9       # nondim ambient air pressure (p_ref = 21 GPa)
    pos = [a for a in argv if not a.startswith("--")]
    if len(pos) >= 1:
        run_dir = Path(pos[0])
    if len(pos) >= 2:
        run_name = pos[1]
    for a in argv:
        if a.startswith("--r0="):
            r0 = float(a.split("=", 1)[1])
        elif a.startswith("--pamb="):
            p_amb = float(a.split("=", 1)[1])
    return run_dir, run_name, r0, p_amb


def snapshot_table(run_dir, run_name):
    pat = re.compile(rf"{re.escape(run_name)}_(\d+)\.h5$")
    out = {}
    for p in sorted(run_dir.glob(f"{run_name}_*.h5")):
        if p.name.endswith(".ckpt.h5"):
            continue
        m = pat.search(p.name)
        if not m:
            continue
        with h5py.File(p, "r") as f:
            t = float(f["time"][0])
        out[int(m.group(1))] = (t, p)
    return out


def grid_radius(n, L, x0):
    dx = L / n
    xc = x0 + (np.arange(n) + 0.5) * dx
    X, Y, Z = np.meshgrid(xc, xc, xc, indexing="ij")
    return np.sqrt(X * X + Y * Y + Z * Z)


def shell_max(field, rflat, bins):
    """Max of `field` within each radius bin."""
    idx = np.digitize(rflat, bins) - 1
    nb = len(bins) - 1
    out = np.full(nb, np.nan)
    valid = (idx >= 0) & (idx < nb)
    fi, ii = field[valid], idx[valid]
    order = np.argsort(ii)
    ii, fi = ii[order], fi[order]
    # reduceat max per bin
    starts = np.searchsorted(ii, np.arange(nb))
    for b in range(nb):
        s = starts[b]
        e = starts[b + 1] if b + 1 < nb else len(ii)
        if e > s:
            out[b] = fi[s:e].max()
    return out


def main():
    run_dir, run_name, r0, p_amb = parse_args(sys.argv[1:])
    table = snapshot_table(run_dir, run_name)
    if not table:
        print(f"No snapshots in {run_dir} (name {run_name})")
        return 1
    steps = sorted(table)

    # Infer grid from the first snapshot.
    with h5py.File(table[steps[0]][1], "r") as f:
        n = f["density"].shape[0]
    L, x0 = 1.0, -0.5
    r = grid_radius(n, L, x0).ravel()
    nb = 96
    bins = np.linspace(0.0, np.sqrt(3.0) * 0.5 * L, nb + 1)
    rmid = 0.5 * (bins[:-1] + bins[1:])
    core = r0 * 1.5     # exclude the products core when finding the leading shock

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # Peak-overpressure envelope: the max pressure ever experienced at each
    # radius is the shock as it passed -> the canonical peak-overpressure-vs-
    # distance curve (robust, no per-snapshot front detection). The front
    # trajectory is the outermost super-ambient radius vs time.
    p_env = np.full(nb, -np.inf)
    shock_thresh = 3.0 * p_amb
    ts, Rs = [], []
    for step in steps:
        t, p = table[step]
        with h5py.File(p, "r") as f:
            pr = np.array(f["pressure"], dtype=np.float64).ravel()
        pmax = shell_max(pr, r, bins)
        good = np.isfinite(pmax)
        # exclude the products core from the propagating-front envelope
        far = good & (rmid > core)
        p_env[far] = np.maximum(p_env[far], pmax[far])
        # leading front = outermost radius with a shock (and inside the box)
        fronts = np.where(far & (rmid < 0.5) & (pmax > shock_thresh))[0]
        if fronts.size:
            ts.append(t); Rs.append(rmid[fronts.max()])
        if step in steps[:: max(1, len(steps) // 6)]:
            axes[2].plot(rmid[good], pmax[good], lw=1.2, label=f"t={t:.3f}")

    ts = np.array(ts); Rs = np.array(Rs)
    env_ok = np.isfinite(p_env) & (rmid < 0.5)
    Z = rmid[env_ok] / r0
    dp = np.maximum(p_env[env_ok] - p_amb, 1e-30)

    # (1) peak overpressure vs scaled distance
    axes[0].loglog(Z, dp, "o-", ms=4)
    if Z.size > 2:
        zref = np.array([Z.min(), Z.max()])
        axes[0].loglog(zref, dp[0] * (zref / zref[0]) ** -3.0, "k--", lw=1,
                       label=r"$Z^{-3}$")
        axes[0].loglog(zref, dp[0] * (zref / zref[0]) ** -1.0, "k:", lw=1,
                       label=r"$Z^{-1}$")
    axes[0].set_xlabel(r"scaled distance $Z = R / r_0$")
    axes[0].set_ylabel(r"peak overpressure $\Delta p$ (nondim, $p_{ref}=p_{CJ}$)")
    axes[0].set_title("Blast decay (peak-pressure envelope)")
    axes[0].legend(fontsize=8); axes[0].grid(True, which="both", alpha=0.3)

    # (2) shock-front trajectory
    axes[1].plot(ts, Rs, "o-", ms=4)
    axes[1].set_xlabel("time (nondim)")
    axes[1].set_ylabel(r"shock-front radius $R_s$")
    axes[1].set_title("Front trajectory")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("radius")
    axes[2].set_ylabel("shell-max pressure")
    axes[2].set_title("Radial pressure profiles")
    axes[2].set_yscale("log")
    axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)

    FIGS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    out_png = FIGS / f"{run_name}_overpressure.png"
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")
    if Rs.size:
        print(f"front snapshots={Rs.size}  R_s: {Rs.min():.3f}->{Rs.max():.3f}")
    print(f"envelope dp over Z=[{Z.min():.2f},{Z.max():.2f}]: "
          f"{dp.max():.3e} -> {dp.min():.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
