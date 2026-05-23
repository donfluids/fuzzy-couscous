"""
Plot shell-averaged energy spectra from a blast_les `<run>_spectra.h5`
file. Overlays multiple times, and optionally splits the solenoidal /
dilatational components.

With --solenoidal the plot reports both axes of a spectrum: the *energy*
(magnitude, via the K_dil/K_tot fraction annotated per time) and the
*turbulence* (spectral shape, via a local-slope panel d log E / d log k).

Usage:
    python tools/plot_spectra.py <spectra.h5> -o <out.png>
        [--solenoidal] [--steps STEPS] [--reference-slope -5/3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Shared spectrum-shape helpers (this script lives in tools/, so spectrum_shape
# is importable directly).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spectrum_shape import (  # noqa: E402
    list_steps, load_step, smooth_slope, integrated_energy,
)


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
    ap.add_argument("--slope-floor", type=float, default=1e-6,
                    help="mask the slope panel where E < this fraction of the "
                         "component peak (trims the roundoff tail). 0 = no mask.")
    ap.add_argument("--slope-smooth", type=float, default=0.4,
                    help="half-width in ln(k) of the slope-panel smoothing fit "
                         "(0 = raw 2-point gradient).")
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

    # --solenoidal adds a second panel for the spectral *shape* (turbulence),
    # alongside the E(k) panel (energy). Without it, a single E(k) panel.
    if args.solenoidal:
        fig, (ax, axs) = plt.subplots(1, 2, figsize=(13, 5))
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
        axs = None

    cmap = plt.get_cmap("viridis")
    for i, step in enumerate(wanted):
        d = load_step(args.spectra, step)
        c = cmap(i / max(1, len(wanted) - 1))
        k = d["k"]
        if args.solenoidal:
            # Energy magnitude: annotate both component fractions (sol + dil = tot).
            Kt = max(integrated_energy(k, d["E_total"]), 1e-30)
            fsol = integrated_energy(k, d["E_sol"]) / Kt
            fdil = integrated_energy(k, d["E_dil"]) / Kt
            label = (f"t={d['time']:.3e} "
                     f"($K_{{sol}}/K_{{tot}}$={fsol:.2f}, "
                     f"$K_{{dil}}/K_{{tot}}$={fdil:.2f})")
            ax.loglog(k[1:], d["E_sol"][1:], "--", color=c, alpha=0.5)
            ax.loglog(k[1:], d["E_dil"][1:], ":", color=c, alpha=0.5)
            # Turbulence shape: per-component local slope. Linestyle encodes the
            # component (solid total, -- sol, : dil), matching the energy panel;
            # color still encodes time.
            for E, ls in ((d["E_total"], "-"), (d["E_sol"], "--"), (d["E_dil"], ":")):
                kk, sl = smooth_slope(k, E, dlnk=args.slope_smooth,
                                      rel_floor=args.slope_floor)
                axs.semilogx(kk, sl, ls, color=c, lw=1.6,
                             alpha=0.9 if ls == "-" else 0.6)
        else:
            label = f"t={d['time']:.3e}"
        ax.loglog(k[1:], d["E_total"][1:], "-", color=c, label=label)

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
        if axs is not None:
            axs.axhline(slope, color="k", ls="--", lw=1,
                        label=f"$k^{{{args.reference_slope}}}$")

    ax.set_xlabel("k")
    ax.set_ylabel("E(k)")
    ax.set_title(f"energy: {args.spectra.name}\n(solid total, -- sol, : dil)"
                 if args.solenoidal else f"{args.spectra.name}")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)

    if axs is not None:
        axs.set_xlabel("k")
        axs.set_ylabel(r"local slope $d\log E/d\log k$")
        axs.set_title("turbulence: spectral slope (solid total, -- sol, : dil)")
        axs.set_ylim(-4, 3)
        axs.legend(fontsize=8, loc="upper right")
        axs.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
