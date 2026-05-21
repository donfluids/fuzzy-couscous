#!/usr/bin/env python3
"""A-priori gate #1: does 1D spherical BHR depend on a hand-tuned b-seed, and how
far is it from the DNS?

The BHR a/b system is (near-)homogeneous in the turbulence variables: the
variable-density (baroclinic) production of a_r is b*dp/dr, and b grows from
a_r*drho/dr -- so without a b-seed the only a-source is the Reynolds-stress term
-(R_rr/rho) drho/dr (which itself needs the hand-seeded k). In 1D spherical
symmetry grad(rho) || grad(p) so there is no baroclinic torque to generate the
fluctuations physically. This script runs the existing 1D BHR
(scripts/bhr_rans_1d.py) over a sweep of b_seed and compares peak a_r, b, k
against the DNS radial profiles (dns_radial_profiles.npz) at t=0.05 and 0.10.

Verdict we expect: 1D under-predicts and/or needs a tuned seed -> motivates 3D.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location("bhr1d", ROOT / "scripts" / "bhr_rans_1d.py")
bhr1d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bhr1d)

B_SEEDS = [0.0, 0.1, 0.5, 1.0, 1.5]
SAVE_T = (0.02, 0.05, 0.10)


def dns_peaks():
    d = np.load(ROOT / "dns_radial_profiles.npz", allow_pickle=True)["profiles"]
    out = {}
    for p in d:
        out[round(float(p["t"]), 2)] = dict(
            k=float(np.nanmax(p["k"])),
            a=float(np.nanmax(np.abs(p["a_r"]))),
            b=float(np.nanmax(p["b"])))
    return out


def main():
    dns = dns_peaks()
    print("DNS peaks:", {t: {kk: f'{vv:.3e}' for kk, vv in v.items()} for t, v in dns.items()})

    results = {}
    for bs in B_SEEDS:
        out = ROOT / f"out_bhr_seed{bs}.npz"
        saves, _ = bhr1d.run(n=400, tend=0.10, use_bhr=True, b_seed=bs,
                             save_times=SAVE_T, out=str(out))
        peaks = {}
        for s in saves:
            t = round(float(s["t"]), 2)
            peaks[t] = dict(k=float(np.max(s["k"])),
                            a=float(np.max(np.abs(s["a"]))),
                            b=float(np.max(s["b"])))
        results[bs] = peaks

    # report ratios RANS/DNS at t=0.05, 0.10
    print("\n  b_seed |  t   | k(RANS/DNS) | |a|(RANS/DNS) | b(RANS/DNS)")
    for bs in B_SEEDS:
        for t in (0.05, 0.10):
            r = results[bs].get(t)
            dt = dns.get(t)
            if r and dt:
                print(f"   {bs:4.1f}  | {t:.2f} | {r['k']/dt['k']:10.2e}  | "
                      f"{r['a']/dt['a']:11.2e}  | {r['b']/max(dt['b'],1e-30):10.2e}")

    # plot: peak a, b, k vs b_seed at t=0.10, with DNS reference
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for j, q in enumerate(["k", "a", "b"]):
        y = [results[bs].get(0.10, {}).get(q, np.nan) for bs in B_SEEDS]
        ax[j].plot(B_SEEDS, y, "o-", color="tab:blue", label="1D BHR")
        if 0.1 in dns:
            ax[j].axhline(dns[0.10][q], color="tab:red", ls="--", label="DNS t=0.10")
        ax[j].set_xlabel("b_seed"); ax[j].set_ylabel(f"peak {q}")
        ax[j].set_yscale("log"); ax[j].set_title(f"peak {q} vs b_seed")
        ax[j].legend(frameon=False); ax[j].grid(True, which="both", ls=":", alpha=0.4)
    fig.suptitle("1D spherical BHR: seed-dependence and gap to DNS (t=0.10)", y=1.02)
    fig.tight_layout()
    out = ROOT / "figs" / "bhr_seed_test.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
