#!/usr/bin/env python3
"""Phase 1c: validate the 3D BHR radial profiles against the LES (DNS) ensemble,
with the 1D BHR overlaid. Reads the solver's bhrprof CSVs (r,k,a_r,b), the DNS
radial profiles (dns_radial_profiles.npz), and the 1D BHR (out_rans_bhr.npz).
Compares k(r), a_r(r), b(r) at t ~ 0.05 and 0.10. figs/bhr3d_vs_dns.png.
"""
import glob
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "out_blast_128_bhr"
TARGETS = [0.05, 0.10]


def load_bhrprof():
    out = []
    for f in sorted(glob.glob(str(RUN / "blast_128_bhr_bhrprof_*.csv"))):
        with open(f) as fh:
            t = float(re.search(r"t=([\d.eE+-]+)", fh.readline()).group(1))
        d = np.genfromtxt(f, delimiter=",", names=True, skip_header=1)
        out.append((t, d["r"], d["k"], d["a_r"], d["b"]))
    return out


def nearest(profs, t):
    return min(profs, key=lambda p: abs(p[0] - t))


def main():
    bhr3d = load_bhrprof()
    if not bhr3d:
        print("no 3D BHR profiles found (run blast_128_bhr.toml first)", file=sys.stderr)
        return 1
    dns = np.load(ROOT / "dns_radial_profiles.npz", allow_pickle=True)["profiles"]
    dns_by_t = {round(float(p["t"]), 2): p for p in dns}
    rans1d = None
    p1 = ROOT / "out_rans_bhr.npz"
    if p1.exists():
        rans1d = list(np.load(p1, allow_pickle=True)["saves"])

    fig, ax = plt.subplots(len(TARGETS), 3, figsize=(15, 4.6*len(TARGETS)))
    if len(TARGETS) == 1:
        ax = ax[None, :]
    for row, tt in enumerate(TARGETS):
        t3, r3, k3, a3, b3 = nearest(bhr3d, tt)
        dkey = min(dns_by_t, key=lambda kk: abs(kk - tt))
        dp = dns_by_t[dkey]
        r1 = a1 = b1 = k1 = None
        if rans1d:
            s = min(rans1d, key=lambda s: abs(float(s["t"]) - tt))
            r1, k1, a1, b1 = s["r"], s["k"], s["a"], s["b"]
        for col, (q3, qd, q1, lab) in enumerate([
                (k3, dp["k"], k1, "k"),
                (a3, dp["a_r"], a1, "a_r"),
                (b3, dp["b"], b1, "b")]):
            a = ax[row, col]
            a.plot(dp["r"], qd, lw=2.4, color="k", label=f"DNS (t={float(dp['t']):.2f})")
            a.plot(r3, q3, lw=2, color="tab:red", label=f"3D BHR (t={t3:.2f})")
            if q1 is not None:
                a.plot(r1, q1, lw=1.4, ls="--", color="tab:blue", label="1D BHR")
            a.set_xlabel("r"); a.set_ylabel(lab)
            a.set_xlim(0, 0.45)
            if lab == "k":
                a.set_yscale("log"); a.set_ylim(1e-3, 1e2)
            a.set_title(f"{lab}(r), t~{tt:.2f}")
            a.legend(frameon=False, fontsize=8); a.grid(True, which="both", ls=":", alpha=0.4)
    fig.suptitle("3D BHR vs DNS (LES) vs 1D BHR: radial turbulence profiles", y=1.0)
    fig.tight_layout()
    out = ROOT / "figs" / "bhr3d_vs_dns.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")

    # quick peak comparison table
    print("\n  t    k_pk(3D / DNS)   |a|_pk(3D / DNS)   b_pk(3D / DNS)")
    for tt in TARGETS:
        t3, r3, k3, a3, b3 = nearest(bhr3d, tt)
        dp = dns_by_t[min(dns_by_t, key=lambda kk: abs(kk - tt))]
        print(f"  {tt:.2f}  {np.nanmax(k3):.2e}/{np.nanmax(dp['k']):.2e}   "
              f"{np.nanmax(np.abs(a3)):.2e}/{np.nanmax(np.abs(dp['a_r'])):.2e}   "
              f"{np.nanmax(b3):.2e}/{np.nanmax(dp['b']):.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
