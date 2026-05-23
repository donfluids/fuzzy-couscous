"""
Diagnose forced-HIT steady state.

Left panel: KE(t), tke(t), eps_total(t), eps_sol(t) time series. After a
few eddy turnovers the dissipation rate should plateau at the user-set
eps_target.

Right panel: averaged shell-energy spectrum E(k) over the last 25% of
the run (the statistically stationary window).

Usage:
    python3 tools/plot_forced_hit.py <out_dir> --eps_target 0.1 -o forced_hit.png
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def list_step_groups(path: Path):
    out = []
    with h5py.File(path, "r") as f:
        for name in f.keys():
            m = re.match(r"step_(\d+)", name)
            if not m:
                continue
            out.append((int(m.group(1)), float(f[name]["time"][0])))
    out.sort()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--eps_target", type=float, default=0.1)
    ap.add_argument("--avg_from", type=float, default=None,
                    help="time to start spectrum averaging (default: 0.75 * t_end)")
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()

    candidates = list(args.out_dir.glob("*_spectra.h5"))
    if not candidates:
        print(f"no *_spectra.h5 in {args.out_dir}", file=sys.stderr)
        return 1
    spectra_path = candidates[0]
    run_name = spectra_path.name.replace("_spectra.h5", "")
    stats_path = args.out_dir / (run_name + "_stats.csv")

    s = pd.read_csv(stats_path, skipinitialspace=True)
    t_end = s.time.iloc[-1]
    avg_from = args.avg_from if args.avg_from is not None else 0.75 * t_end

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: KE / tke / eps time series.
    axL = axes[0]
    axL.plot(s.time, s.tke, label="tke (= 1/2 <u'^2> per mass)", color="C0")
    axL.set_ylabel("tke", color="C0")
    axL.tick_params(axis="y", labelcolor="C0")
    axL.set_xlabel("time")
    axL.grid(True, alpha=0.3)

    axR = axL.twinx()
    axR.plot(s.time, s.eps_total, label=r"$\varepsilon_{tot}$", color="C3")
    axR.plot(s.time, s.eps_sol, "--", label=r"$\varepsilon_{sol}$", color="C1")
    axR.axhline(args.eps_target, color="k", linestyle=":",
                label=fr"$\varepsilon_{{target}} = {args.eps_target:.2f}$")
    axR.set_ylabel("dissipation rate")
    axR.tick_params(axis="y", labelcolor="C3")

    h1, l1 = axL.get_legend_handles_labels()
    h2, l2 = axR.get_legend_handles_labels()
    axR.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
    axL.set_title(f"{run_name}: forced HIT energy balance")

    # Right: spectra averaged over the statistically stationary window.
    steps_times = list_step_groups(spectra_path)
    sel = [(st, tt) for (st, tt) in steps_times if tt >= avg_from]
    if not sel:
        sel = steps_times[-5:]
    print(f"  averaging {len(sel)} spectra over t in [{sel[0][1]:.2f}, "
          f"{sel[-1][1]:.2f}]")

    with h5py.File(spectra_path, "r") as f:
        k_arr = f[f"step_{sel[0][0]:06d}"]["k"][...]
        avg_E = np.zeros_like(k_arr)
        for (step, _) in sel:
            avg_E += f[f"step_{step:06d}"]["E_total"][...]
        avg_E /= len(sel)

    axS = axes[1]
    axS.loglog(k_arr[1:], avg_E[1:], "-", color="C0", lw=1.5,
               label=f"E(k), avg t∈[{sel[0][1]:.1f}, {sel[-1][1]:.1f}]")

    # Reference Kolmogorov -5/3 line.
    good = (k_arr > 2) & (k_arr < 8)
    if np.any(good):
        idx = int(np.where(good)[0][len(np.where(good)[0]) // 2])
        anchor_k = float(k_arr[idx])
        anchor_E = float(avg_E[idx])
        k_ref = np.geomspace(2.0, 12.0, 32)
        ref = anchor_E * (k_ref / anchor_k) ** (-5.0 / 3.0)
        axS.loglog(k_ref, ref, "k--", lw=1.2, label=r"$\propto k^{-5/3}$")

    axS.set_xlabel("wavenumber k")
    axS.set_ylabel("E(k)")
    axS.set_title("Time-averaged spectrum (stationary window)")
    axS.grid(True, which="both", alpha=0.3)
    axS.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
