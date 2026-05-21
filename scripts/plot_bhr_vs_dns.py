#!/usr/bin/env python3
"""Overlay 1D BHR RANS radial profiles against DNS spherical+ensemble averages.

RANS:  out_rans_bhr.npz  (saves: t, r, rho, u, p, k, eps, a, b)
DNS:   dns_radial_profiles.npz (profiles: t, r, rho, u_r, p, k, a_r, b)

Produces figs/bhr_vs_dns.png: a 2x3 grid (rho, u_r, p, k, a_r, b), each panel
overlaying RANS (solid) and DNS (dashed) at the comparison times, colored by
time. Comparison is cleanest in the spherical phase t <~ 0.1.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
DATA = ROOT / "data"

rans = np.load(DATA / "out_rans_bhr.npz", allow_pickle=True)
rans_saves = list(rans["saves"])
dns = np.load(DATA / "dns_radial_profiles.npz", allow_pickle=True)
dns_profiles = list(dns["profiles"])

# Times to compare (use DNS times; match nearest RANS save).
cmp_times = [p["t"] for p in dns_profiles if p["t"] <= 0.26]
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(cmp_times)))


def nearest_rans(t):
    return min(rans_saves, key=lambda s: abs(float(s["t"]) - t))


def nearest_dns(t):
    return min(dns_profiles, key=lambda s: abs(float(s["t"]) - t))


panels = [
    ("rho",  "rho",  r"$\bar\rho$",      "linear"),
    ("u",    "u_r",  r"$\bar u_r$",      "linear"),
    ("p",    "p",    r"$\bar p$",        "linear"),
    ("k",    "k",    r"$k$",             "log"),
    ("a",    "a_r",  r"$a_r$",           "linear"),
    ("b",    "b",    r"$b$",             "linear"),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.ravel()

for ax, (rkey, dkey, label, yscale) in zip(axes, panels):
    for c, t in zip(colors, cmp_times):
        rs = nearest_rans(t)
        ds = nearest_dns(t)
        rr = rs["r"]; rv = np.asarray(rs[rkey], dtype=float)
        dr = ds["r"]; dv = np.asarray(ds[dkey], dtype=float)
        if yscale == "log":
            rv = np.maximum(rv, 1e-12); dv = np.maximum(dv, 1e-12)
        ax.plot(rr, rv, "-", color=c, lw=1.8,
                label=f"RANS t={float(rs['t']):.3f}")
        ax.plot(dr, dv, "--", color=c, lw=1.5,
                label=f"DNS t={float(ds['t']):.3f}")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(label)
    ax.set_title(label)
    if yscale == "log":
        ax.set_yscale("log")
    ax.set_xlim(0, 0.5)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)

# Legend only on first panel to avoid clutter.
axes[0].legend(frameon=False, fontsize=7, ncol=2, loc="best")

fig.suptitle(
    "1D BHR RANS (solid) vs DNS spherical+ensemble average (dashed)\n"
    "blast, slip-wall chamber; spherical phase t < 0.1 is the clean comparison",
    y=1.02, fontsize=12)
out = ROOT / "figs" / "bhr_vs_dns.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"Wrote {out}")

# Also a KE(t) comparison (RANS resolved KE vs DNS resolved KE).
fig2, ax = plt.subplots(figsize=(7, 5))
ke = rans["ke_hist"]
ax.plot(ke[:, 0], ke[:, 1], "-", color="tab:blue", lw=1.8, label="RANS KE (mean flow)")
ax.plot(ke[:, 0], ke[:, 2], "-", color="tab:orange", lw=1.8, label="RANS TKE (turb)")
# DNS volume-integrated KE / TKE from the budget npz times (approx).
import csv
dt, dKE = [], []
with open(RUNS / "out_blast_128_budget_seed1" /
          "blast_128_budget_seed1_stats.csv") as f:
    for row in csv.DictReader(f):
        dt.append(float(row["time"])); dKE.append(float(row["KE"]))
ax.plot(dt, dKE, "k--", lw=1.4, label="DNS KE (one seed)")
ax.set_xlabel(r"$t$"); ax.set_ylabel("kinetic energy (volume integral)")
ax.set_title("KE(t): RANS vs DNS")
ax.set_xlim(0, 0.5)
ax.legend(frameon=False)
ax.grid(True, ls=":", alpha=0.5)
out2 = ROOT / "figs" / "bhr_vs_dns_ke.png"
fig2.tight_layout()
fig2.savefig(out2, dpi=140, bbox_inches="tight")
print(f"Wrote {out2}")
