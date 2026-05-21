#!/usr/bin/env python3
"""Calibration sweep for the 1D BHR RANS against DNS radial profiles.

Runs the BHR model with several constant sets, scores each against the DNS
peak turbulence quantities (k, |a_r|, b) at t~0.05, and the TKE level.
Score = sum of |log10(RANS_peak / DNS_peak)| over {k, |a|, b}; lower is better.
"""

from pathlib import Path
import numpy as np

import importlib.util
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
spec = importlib.util.spec_from_file_location(
    "bhr1d", ROOT / "scripts" / "bhr_rans_1d.py")
bhr1d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bhr1d)

# DNS targets at t~0.05 (from dns_radial_average.py output).
dns = np.load(DATA / "dns_radial_profiles.npz", allow_pickle=True)
dns_profiles = list(dns["profiles"])
dref = min(dns_profiles, key=lambda p: abs(float(p["t"]) - 0.05))
dns_k = float(np.max(dref["k"]))
dns_a = float(np.max(np.abs(dref["a_r"])))
dns_b = float(np.max(dref["b"]))
print(f"DNS targets @ t={float(dref['t']):.3f}: "
      f"k_peak={dns_k:.3e}  |a|_peak={dns_a:.3e}  b_peak={dns_b:.3e}\n")

# With b seeded at the contact (bootstraps a_r via b*dp/dr, since 1D spherical
# has no baroclinic generation). Sweep b_seed and the destruction constants.
CONFIGS = [
    dict(name="bseed0.1",        prod_limit=1e3, c_a=1.0, c_b=1.0, c_mu=0.09, seed_scale=1e-3, b_seed=0.1, feedback=False),
    dict(name="bseed0.3",        prod_limit=1e3, c_a=1.0, c_b=1.0, c_mu=0.09, seed_scale=1e-3, b_seed=0.3, feedback=False),
    dict(name="bseed0.5",        prod_limit=1e3, c_a=1.0, c_b=1.0, c_mu=0.09, seed_scale=1e-3, b_seed=0.5, feedback=False),
    dict(name="bseed0.5+lowd",   prod_limit=1e3, c_a=0.3, c_b=0.3, c_mu=0.09, seed_scale=1e-3, b_seed=0.5, feedback=False),
    dict(name="bseed1.0+lowd",   prod_limit=1e3, c_a=0.3, c_b=0.3, c_mu=0.09, seed_scale=1e-3, b_seed=1.0, feedback=False),
    dict(name="bseed1.5+lowd",   prod_limit=1e3, c_a=0.3, c_b=0.3, c_mu=0.09, seed_scale=1e-3, b_seed=1.5, feedback=False),
    dict(name="bseed1.5+vlowd",  prod_limit=1e3, c_a=0.1, c_b=0.1, c_mu=0.09, seed_scale=1e-3, b_seed=1.5, feedback=False),
]


def score(cfg):
    out = DATA / f"out_cal_{cfg['name']}.npz"
    saves, ke = bhr1d.run(
        n=300, rwall=0.5, tend=0.12, use_bhr=True,
        feedback=cfg["feedback"], c_mu=cfg["c_mu"], c_a=cfg["c_a"],
        c_b=cfg["c_b"], prod_limit=cfg["prod_limit"],
        seed_scale=cfg["seed_scale"], b_seed=cfg.get("b_seed", 0.0),
        save_times=(0.05, 0.1), out=str(out))
    s = min(saves, key=lambda x: abs(x["t"] - 0.05))
    k_pk = float(np.max(s["k"]))
    a_pk = float(np.max(np.abs(s["a"])))
    b_pk = float(np.max(s["b"]))
    def lr(x, ref):
        return abs(np.log10(max(x, 1e-12) / max(ref, 1e-12)))
    sc = lr(k_pk, dns_k) + lr(a_pk, dns_a) + lr(b_pk, dns_b)
    return k_pk, a_pk, b_pk, sc


print(f"{'config':18s} {'k_peak':>10s} {'|a|_peak':>10s} {'b_peak':>10s} {'score':>8s}")
results = []
for cfg in CONFIGS:
    try:
        k_pk, a_pk, b_pk, sc = score(cfg)
        results.append((cfg["name"], k_pk, a_pk, b_pk, sc))
        print(f"{cfg['name']:18s} {k_pk:10.3e} {a_pk:10.3e} {b_pk:10.3e} {sc:8.3f}")
    except Exception as e:
        print(f"{cfg['name']:18s} FAILED: {e}")

if results:
    best = min(results, key=lambda r: r[4])
    print(f"\nBest config: {best[0]} (score {best[4]:.3f})")
    print(f"  vs DNS: k {best[1]:.2e}/{dns_k:.2e}, "
          f"|a| {best[2]:.2e}/{dns_a:.2e}, b {best[3]:.2e}/{dns_b:.2e}")
