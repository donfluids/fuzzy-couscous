#!/usr/bin/env python3
"""Summarise the INITIAL-CONDITION differences between the single-gas (SG) and
two-gamma multifluid (MF) blast.

Both are initialised IDENTICALLY in (rho, p, T) and the Y42 interface seed; the
ONLY difference is the equation of state in the products blob:
  SG: gamma = 1.4 everywhere.
  MF: products gamma_p = 1.25 (denser, larger cv) vs air gamma = 1.4, carried by
      an advected G = 1/(gamma-1) field -> a persistent material/gamma interface.
Consequences at fixed (rho, p): MF stores MORE internal energy e = p/(gamma-1)
(G_p=4 vs G_air=2.5 -> 1.6x) and has a LOWER sound speed c = sqrt(gamma p/rho).

NOTE on snapshot pressure: HDF5Writer writes p with the GLOBAL gamma, so for MF
the saved 'pressure' is (gamma_air-1)*e, not the true (gamma_local-1)*e. At t=0
velocity=0 so E = e_int = p_snapshot/(gamma_air-1) is exact for both; we then
rebuild the TRUE p = e_int/G with G from the density-derived interface weight.
"""
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
N = sys.argv[1] if len(sys.argv) > 1 else "128"
GA, GP = 2.5, 4.0           # 1/(gamma-1) for air (1.4) and products (1.25)
RHO_A, RHO_P = 1.0, 17.96   # ambient / CJ products density
R = 1.0

def load(rel):
    with h5py.File(run_dir(rel), "r") as f:
        return np.asarray(f["density"]), np.asarray(f["pressure"])

rho_mf, ps_mf = load(f"out_blast_{N}_cj_t5/blast_{N}_cj_t5_000000.h5")
rho_sg, ps_sg = load(f"out_blast_{N}_sg_t5/blast_{N}_sg_t5_000000.h5")
n = rho_mf.shape[0]; dx = 1.0 / n
# internal-energy density E (exact for both at t=0; undo the writer's gamma_air-1)
e_mf = ps_mf * GA
e_sg = ps_sg * GA
# interface weight from density; G and true fields
w_mf = np.clip((rho_mf - RHO_A) / (RHO_P - RHO_A), 0, 1)
G_mf = GA + (GP - GA) * w_mf
gam_mf = 1.0 + 1.0 / G_mf
p_mf = e_mf / G_mf
c_mf = np.sqrt(gam_mf * p_mf / rho_mf)
G_sg = np.full_like(rho_sg, GA)
gam_sg = np.full_like(rho_sg, 1.4)
p_sg = e_sg / GA
c_sg = np.sqrt(1.4 * p_sg / rho_sg)

# radial (spherical) averages
xc = (np.arange(n) + 0.5) * dx - 0.5
X, Y, Z = np.meshgrid(xc, xc, xc, indexing="ij")
rr = np.sqrt(X**2 + Y**2 + Z**2)
nb = 64; redge = np.linspace(0, 0.45, nb + 1); rc = 0.5 * (redge[1:] + redge[:-1])
idx = np.clip(np.digitize(rr.ravel(), redge) - 1, 0, nb - 1)
cnt = np.bincount(idx, minlength=nb).astype(float); cnt[cnt == 0] = 1
def radavg(a): return np.bincount(idx, a.ravel(), minlength=nb) / cnt

mid = n // 2
fig = plt.figure(figsize=(14, 9))

# --- 2D slices (z=mid): density (shared) and gamma (MF) ---
ax = fig.add_subplot(2, 3, 1)
im = ax.imshow(rho_mf[mid], origin="lower", extent=[-.5, .5, -.5, .5], cmap="viridis")
ax.set_title(r"$\rho$ at $t=0$  (IDENTICAL: SG = MF)"); plt.colorbar(im, ax=ax, shrink=.8)
ax.set_xlabel("x"); ax.set_ylabel("y")

ax = fig.add_subplot(2, 3, 2)
im = ax.imshow(gam_mf[mid], origin="lower", extent=[-.5, .5, -.5, .5], cmap="coolwarm")
ax.set_title(r"MF $\gamma$ at $t=0$ (products 1.25 vs air 1.4)"); plt.colorbar(im, ax=ax, shrink=.8)
ax.set_xlabel("x"); ax.set_ylabel("y")

ax = fig.add_subplot(2, 3, 3)
im = ax.imshow(e_mf[mid] - e_sg[mid], origin="lower", extent=[-.5, .5, -.5, .5], cmap="magma")
ax.set_title(r"internal energy excess  $e_{MF}-e_{SG}$"); plt.colorbar(im, ax=ax, shrink=.8)
ax.set_xlabel("x"); ax.set_ylabel("y")

# --- radial profiles: matched (rho, p) ---
ax = fig.add_subplot(2, 3, 4)
ax.plot(rc, radavg(rho_mf), "r-", lw=2.5, label=r"$\rho$ MF")
ax.plot(rc, radavg(rho_sg), "b--", lw=1.5, label=r"$\rho$ SG")
ax.set_xlabel("r"); ax.set_ylabel(r"$\rho$", color="k")
ax2 = ax.twinx()
ax2.plot(rc, radavg(p_mf), color="darkred", lw=2.5, ls=":", label=r"$p$ MF")
ax2.plot(rc, radavg(p_sg), color="navy", lw=1.5, ls="-.", label=r"$p$ SG")
ax2.set_ylabel(r"$p$ (true)")
ax.set_title(r"$\rho$ and $p$: MATCHED (SG $\equiv$ MF)")
ax.legend(loc="upper right", fontsize=8); ax2.legend(loc="center right", fontsize=8)

# --- radial profiles: gamma (differs) ---
ax = fig.add_subplot(2, 3, 5)
ax.plot(rc, radavg(gam_mf), "r-", lw=2.5, label=r"$\gamma$ MF")
ax.plot(rc, radavg(gam_sg), "b--", lw=2, label=r"$\gamma$ SG (=1.4)")
ax.set_xlabel("r"); ax.set_ylabel(r"$\gamma$"); ax.set_ylim(1.2, 1.45)
ax.set_title(r"$\gamma$: DIFFERS (the only IC knob)")
ax.legend(fontsize=9); ax.grid(True, ls=":", alpha=0.4)

# --- radial profiles: internal energy + sound speed (differ) ---
ax = fig.add_subplot(2, 3, 6)
ax.plot(rc, radavg(e_mf), "r-", lw=2.5, label=r"$e$ MF")
ax.plot(rc, radavg(e_sg), "b--", lw=1.8, label=r"$e$ SG")
ax.set_xlabel("r"); ax.set_ylabel(r"internal energy $e=p/(\gamma-1)$", color="k")
ax3 = ax.twinx()
ax3.plot(rc, radavg(c_mf), color="darkred", lw=2, ls=":", label=r"$c$ MF")
ax3.plot(rc, radavg(c_sg), color="navy", lw=1.5, ls="-.", label=r"$c$ SG")
ax3.set_ylabel(r"sound speed $c=\sqrt{\gamma p/\rho}$")
emf = radavg(e_mf).max(); esg = radavg(e_sg).max()
ax.set_title(f"$e$ DIFFERS ({emf/esg:.2f}$\\times$ in blob), $c$ lower in MF")
ax.legend(loc="upper right", fontsize=8); ax3.legend(loc="center right", fontsize=8)

fig.suptitle(f"Initial conditions: single-gas (SG) vs two-gamma multifluid (MF), "
             f"{N}$^3$ — matched $\\rho,p,T$; only $\\gamma$ (hence $e$, $c$) differ",
             y=1.0, fontsize=12)
fig.tight_layout()
out = ROOT / "figs" / f"ic_compare_mf_sg_{N}.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
print(f"blob: e_MF/e_SG = {emf/esg:.3f}; gamma MF_blob={gam_mf.min():.3f} vs SG=1.4; "
      f"c_MF_blob/c_SG_blob = {radavg(c_mf)[:5].mean()/radavg(c_sg)[:5].mean():.3f}")
