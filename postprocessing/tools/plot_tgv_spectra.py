"""
Overlay TGV-Re1600 spectra at multiple times and check for emergence of the
Kolmogorov k^{-5/3} inertial range.

Usage:
    python3 tools/plot_tgv_spectra.py <out_dir> -o spectra.png
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


def load_spectrum(path: Path, step: int):
    with h5py.File(path, "r") as f:
        g = f[f"step_{step:06d}"]
        return {
            "k": g["k"][...],
            "E_total": g["E_total"][...],
            "E_sol":   g["E_sol"][...],
            "E_dil":   g["E_dil"][...],
            "time":    float(g["time"][0]),
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path,
                    help="run output directory containing <run>_spectra.h5 and <run>_stats.csv")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--times", type=str, default="0.5,2,4,6,8,10",
                    help="comma-separated target times to plot")
    args = ap.parse_args()

    # Locate the spectra HDF5 (one file ending in _spectra.h5).
    candidates = list(args.out_dir.glob("*_spectra.h5"))
    if not candidates:
        print(f"no *_spectra.h5 in {args.out_dir}", file=sys.stderr)
        return 1
    spectra_path = candidates[0]
    run_name = spectra_path.name.replace("_spectra.h5", "")
    stats_path = args.out_dir / (run_name + "_stats.csv")

    steps_times = list_step_groups(spectra_path)
    if not steps_times:
        print(f"no step groups in {spectra_path}", file=sys.stderr)
        return 1
    print(f"  {len(steps_times)} spectra in {spectra_path.name}, "
          f"t range [{steps_times[0][1]:.3f}, {steps_times[-1][1]:.3f}]")

    want_times = [float(s) for s in args.times.split(",")]
    selected = []
    for tt in want_times:
        # Nearest available
        best = min(steps_times, key=lambda st: abs(st[1] - tt))
        selected.append(best)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    cmap = plt.get_cmap("plasma")
    for i, (step, t) in enumerate(selected):
        d = load_spectrum(spectra_path, step)
        color = cmap(i / max(1, len(selected) - 1))
        k = d["k"][1:]
        E = d["E_total"][1:]
        axes[0].loglog(k, E, "-", color=color, label=f"t = {t:.2f}")

        # Local slope d log E / d log k via central differences on log-log.
        good = E > 0
        kl = np.log(k[good]); El = np.log(E[good])
        if len(kl) > 4:
            slope = np.gradient(El, kl)
            axes[1].semilogx(k[good], slope, "-", color=color,
                             label=f"t = {t:.2f}")

    # Reference Kolmogorov -5/3 line on the spectrum plot.
    d_late = load_spectrum(spectra_path, selected[-1][0])
    k_arr = d_late["k"][1:]
    E_late = d_late["E_total"][1:]
    # Anchor in the inertial-range candidate band around k ~ 5-6.
    idx = np.argmin(np.abs(k_arr - 6.0))
    anchor_k = k_arr[idx]
    anchor_E = E_late[idx]
    k_ref = np.geomspace(3.0, 18.0, 32)
    ref_line = anchor_E * (k_ref / anchor_k) ** (-5.0 / 3.0)
    axes[0].loglog(k_ref, ref_line, "k--", lw=1.5, label=r"$\propto k^{-5/3}$")

    axes[0].set_xlabel("wavenumber k")
    axes[0].set_ylabel("E(k)")
    axes[0].set_title(f"{run_name}: TGV cascade spectra")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="lower left", fontsize=8, ncol=2)

    # Local-slope diagnostic.
    axes[1].axhline(-5.0 / 3.0, color="k", linestyle="--", lw=1,
                    label=r"$-5/3$")
    axes[1].set_xlabel("wavenumber k")
    axes[1].set_ylabel(r"local slope $\,d \log E / d \log k$")
    axes[1].set_title("Local spectral slope")
    axes[1].set_ylim(-5, 2)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower left", ncol=2)

    # Right panel: integrated quantities vs time.
    if stats_path.exists():
        s = pd.read_csv(stats_path, skipinitialspace=True)
        ax = axes[2]
        ax.plot(s["time"], s["tke"], label="tke", color="C0")
        ax.set_xlabel("time")
        ax.set_ylabel("tke", color="C0")
        ax.tick_params(axis="y", labelcolor="C0")
        ax2 = ax.twinx()
        ax2.plot(s["time"], s["eps_total"], label="$\\varepsilon_{tot}$", color="C3")
        ax2.set_ylabel("dissipation rate", color="C3")
        ax2.tick_params(axis="y", labelcolor="C3")
        for _, t in selected:
            ax.axvline(t, color="gray", alpha=0.2, linestyle=":")
        ax.set_title("KE decay and dissipation")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
