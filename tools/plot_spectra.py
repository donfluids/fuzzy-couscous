"""
Plot shell-averaged energy spectra from a blast_les `<run>_spectra.h5`
file. Overlays multiple times, and optionally splits the solenoidal /
dilatational components.

Usage:
    python tools/plot_spectra.py <spectra.h5> -o <out.png>
        [--solenoidal] [--steps STEPS] [--reference-slope -5/3]
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


def list_steps(path: Path) -> list[tuple[int, float]]:
    out = []
    with h5py.File(path, "r") as f:
        for name in f.keys():
            m = re.match(r"step_(\d+)", name)
            if not m:
                continue
            step = int(m.group(1))
            t = float(f[name]["time"][0])
            out.append((step, t))
    out.sort()
    return out


def load_step(path: Path, step: int) -> dict[str, np.ndarray]:
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
    ap.add_argument("spectra", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--steps", type=str, default=None,
                    help="comma-separated step indices to plot; default = "
                         "evenly spaced subsample of all available steps")
    ap.add_argument("--solenoidal", action="store_true",
                    help="also plot solenoidal/dilatational components")
    ap.add_argument("--reference-slope", default="-5/3",
                    help="reference slope (e.g. -5/3, -2). Set to 'none' to omit.")
    args = ap.parse_args()

    available = list_steps(args.spectra)
    if not available:
        print(f"no step_NNNNNN groups in {args.spectra}", file=sys.stderr)
        return 1

    if args.steps:
        wanted = [int(s) for s in args.steps.split(",")]
    else:
        n_show = min(6, len(available))
        idx = np.linspace(0, len(available) - 1, n_show).astype(int)
        wanted = [available[i][0] for i in idx]

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("viridis")
    for i, step in enumerate(wanted):
        d = load_step(args.spectra, step)
        c = cmap(i / max(1, len(wanted) - 1))
        ax.loglog(d["k"][1:], d["E_total"][1:], "-", color=c,
                  label=f"t={d['time']:.3e}")
        if args.solenoidal:
            ax.loglog(d["k"][1:], d["E_sol"][1:], "--", color=c, alpha=0.5)
            ax.loglog(d["k"][1:], d["E_dil"][1:], ":", color=c, alpha=0.5)

    if args.reference_slope and args.reference_slope != "none":
        if "/" in args.reference_slope:
            num, den = args.reference_slope.split("/")
            slope = float(num) / float(den)
        else:
            slope = float(args.reference_slope)
        kk = np.logspace(np.log10(2), np.log10(20), 50)
        anchor = 10
        ax.loglog(kk, anchor * (kk / kk[0]) ** slope, "k--", lw=1,
                  label=f"$\\propto k^{{{args.reference_slope}}}$")

    ax.set_xlabel("k")
    ax.set_ylabel("E(k)")
    ax.set_title(f"{args.spectra.name}")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
