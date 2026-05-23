#!/usr/bin/env python3
"""Movie of the JWL TNT blast (closed chamber) on a z=mid slice.

Three panels following the physics we care about:
  density   (log) : dense detonation products expanding into ambient air; the
                    Y42-seeded contact wrinkling + mixing (Atwood ~ 0.999).
  pressure  (log) : the lead shock and its wall reflections / re-shocks.
  |omega|         : vorticity magnitude -- the SOLENOIDAL turbulence, pumped by
                    baroclinic torque + Richtmyer-Meshkov at each re-shock.

Usage: python scripts/movie_tnt.py [run_dir] [run_name]
Default: solver/runs/out_tnt_chamber_128 / tnt_chamber_128
"""
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LogNorm

import sys
from pathlib import Path
_PP = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").is_file())
for _d in (_PP, _PP / "tools", _PP / "scripts"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
from paths import REPO_ROOT as ROOT, RUNS, DATA, FIGS, run_dir  # noqa: E402
run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    run_dir("out_tnt_chamber_128")
name = sys.argv[2] if len(sys.argv) > 2 else "tnt_chamber_128"

pat = re.compile(rf"{re.escape(name)}_(\d+)\.h5$")
snaps = sorted([p for p in run_dir.glob(f"{name}_*.h5")
                if pat.search(p.name) and not p.name.endswith("_spectra.h5")],
               key=lambda p: int(pat.search(p.name).group(1)))
print(f"{len(snaps)} frames from {run_dir.name}")
if not snaps:
    sys.exit(1)


def load(p):
    with h5py.File(p, "r") as f:
        return (float(f["time"][0]), np.asarray(f["density"]),
                np.asarray(f["pressure"]), np.asarray(f["velocity_x"]),
                np.asarray(f["velocity_y"]), np.asarray(f["velocity_z"]))


def vort_mag(u, v, w, dx):
    uy, uz = np.gradient(u, dx, axis=1), np.gradient(u, dx, axis=0)
    vx, vz = np.gradient(v, dx, axis=2), np.gradient(v, dx, axis=0)
    wx, wy = np.gradient(w, dx, axis=2), np.gradient(w, dx, axis=1)
    ox, oy, oz = wy - vz, uz - wx, vx - uy
    return np.sqrt(ox * ox + oy * oy + oz * oz)


# Fixed color scales from a late-time frame (turbulence well developed).
tm, rho_m, p_m, um, vm, wm = load(snaps[2 * len(snaps) // 3])
n = rho_m.shape[0]; dx = 1.0 / n; mid = n // 2
wmax = np.percentile(vort_mag(um, vm, wm, dx)[mid], 99.0)
RHO_VMIN, RHO_VMAX = 5e-4, 1.5
P_VMIN, P_VMAX = 1e-5, 1.0

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.4))
ext = [-.5, .5, -.5, .5]
imD = ax[0].imshow(rho_m[mid], origin="lower", extent=ext, cmap="turbo",
                   norm=LogNorm(vmin=RHO_VMIN, vmax=RHO_VMAX))
ax[0].set_title("density"); plt.colorbar(imD, ax=ax[0], shrink=0.8, label=r"$\rho$ (log)")
imP = ax[1].imshow(p_m[mid], origin="lower", extent=ext, cmap="magma",
                   norm=LogNorm(vmin=P_VMIN, vmax=P_VMAX))
ax[1].set_title("pressure"); plt.colorbar(imP, ax=ax[1], shrink=0.8, label=r"$p$ (log)")
imW = ax[2].imshow(vort_mag(um, vm, wm, dx)[mid], origin="lower", extent=ext,
                   cmap="inferno", vmin=0, vmax=wmax)
ax[2].set_title(r"vorticity magnitude $|\omega|$ (solenoidal)")
plt.colorbar(imW, ax=ax[2], shrink=0.8, label=r"$|\omega|$")
for a in ax:
    a.set_xlabel("x"); a.set_ylabel("y")
sup = fig.suptitle("", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])

out = ROOT / "figs" / "tnt" / f"movie_{name}.mp4"
out.parent.mkdir(parents=True, exist_ok=True)
writer = FFMpegWriter(fps=4, bitrate=3000)
with writer.saving(fig, str(out), dpi=130):
    for p in snaps:
        t, rho, pr, u, v, w = load(p)
        imD.set_data(rho[mid]); imP.set_data(pr[mid])
        imW.set_data(vort_mag(u, v, w, dx)[mid])
        sup.set_text(f"JWL TNT blast, closed chamber, {n}$^3$    t = {t:.3f}")
        writer.grab_frame()
        print(f"  frame t={t:.3f}")
print(f"wrote {out}")
