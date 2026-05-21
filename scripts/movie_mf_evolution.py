#!/usr/bin/env python3
"""Movie of the two-gamma multifluid (MF) blast evolving from its initial state.
Left  : density slice (z=mid) -- dense products expanding into light air, the
        Y42-seeded interface rolling up and mixing.
Right : vorticity magnitude |omega| (z=mid) -- where baroclinic turbulence is born.
Usage: movie_mf_evolution.py [N]   (N=64 or 128, default 128)
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

ROOT = Path(__file__).resolve().parent.parent
RUNROOT = ROOT / "solver"
N = sys.argv[1] if len(sys.argv) > 1 else "128"
run = f"out_blast_{N}_cj_t5"; name = f"blast_{N}_cj_t5"
rd = RUNROOT / run

pat = re.compile(rf"{re.escape(name)}_(\d+)\.h5$")
snaps = sorted([p for p in rd.glob(f"{name}_*.h5")
                if pat.search(p.name) and not p.name.endswith("_spectra.h5")],
               key=lambda p: int(pat.search(p.name).group(1)))
print(f"{len(snaps)} frames from {run}")

def load(p):
    with h5py.File(p, "r") as f:
        return (float(f["time"][0]), np.asarray(f["density"]),
                np.asarray(f["velocity_x"]), np.asarray(f["velocity_y"]),
                np.asarray(f["velocity_z"]))

def vort_mag(u, v, w, dx):
    # |omega| is invariant under the [z,y,x] storage order; compute on raw arrays.
    uy, uz = np.gradient(u, dx, axis=1), np.gradient(u, dx, axis=0)
    vx, vz = np.gradient(v, dx, axis=2), np.gradient(v, dx, axis=0)
    wx, wy = np.gradient(w, dx, axis=2), np.gradient(w, dx, axis=1)
    ox, oy, oz = wy - vz, uz - wx, vx - uy
    return np.sqrt(ox*ox + oy*oy + oz*oz)

# fixed color scales (use a mid-time frame for the vorticity ceiling)
n = None
tmid, rho_m, um, vm, wm = load(snaps[len(snaps)//3])
n = rho_m.shape[0]; dx = 1.0 / n; mid = n // 2
wmax = np.percentile(vort_mag(um, vm, wm, dx)[mid], 99.5)
RHO_VMIN, RHO_VMAX = 0.3, 18.0

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.6))
ext = [-.5, .5, -.5, .5]
imL = axL.imshow(rho_m[mid], origin="lower", extent=ext, cmap="turbo",
                 norm=LogNorm(vmin=RHO_VMIN, vmax=RHO_VMAX))
axL.set_title("density"); axL.set_xlabel("x"); axL.set_ylabel("y")
plt.colorbar(imL, ax=axL, shrink=0.85, label=r"$\rho$ (log)")
imR = axR.imshow(vort_mag(um, vm, wm, dx)[mid], origin="lower", extent=ext,
                 cmap="inferno", vmin=0, vmax=wmax)
axR.set_title(r"vorticity magnitude $|\omega|$"); axR.set_xlabel("x"); axR.set_ylabel("y")
plt.colorbar(imR, ax=axR, shrink=0.85, label=r"$|\omega|$")
sup = fig.suptitle("", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])

out = ROOT / "figs" / f"movie_mf_evolution_{N}.mp4"
writer = FFMpegWriter(fps=4, bitrate=2400)
with writer.saving(fig, str(out), dpi=130):
    for p in snaps:
        t, rho, u, v, w = load(p)
        imL.set_data(rho[mid])
        imR.set_data(vort_mag(u, v, w, dx)[mid])
        sup.set_text(f"Two-gamma multifluid blast (products $\\gamma$=1.25 into air $\\gamma$=1.4), "
                     f"{N}$^3$    t = {t:.3f}")
        writer.grab_frame()
        print(f"  frame t={t:.3f}")
print(f"wrote {out}")
