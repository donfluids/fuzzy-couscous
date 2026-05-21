#!/usr/bin/env python3
"""Does a SMALL averaging window recover k^-5/3 in the TOTAL spectrum?

Scans narrow time-average windows across a dense single-seed run and, for each,
reports the slope of the total spectrum E_total(k) in the post-peak
inertial-candidate range, plus the solenoidal/dilatational slopes for reference.
The idea: a wide [0.15,0.5] average smears an evolving spectrum; a small window
is a cleaner test of whether the instantaneous total spectrum is ~ -5/3.

Total spectra rise to a low-k acoustic peak then fall, so the slope is fit over
the FALLING range (default k in [12.6, 63], i.e. just above the fundamental to
mid-k, below the dissipation roll-off).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spectrum_components import spectra_path, tavg, local_slope  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def fit_slope(k, E, klo, khi):
    m = (k >= klo) & (k <= khi) & (E > 0)
    if m.sum() < 4:
        return np.nan
    return np.polyfit(np.log(k[m]), np.log(E[m]), 1)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="first05")
    ap.add_argument("--t-end", type=float, default=0.05)
    ap.add_argument("--width", type=float, default=0.01, help="window width")
    ap.add_argument("--step", type=float, default=0.01, help="window stride")
    ap.add_argument("--kfit", type=float, nargs=2, default=(12.6, 63.0),
                    help="k range for the slope fit (falling/inertial region)")
    args = ap.parse_args()

    path = spectra_path(args.tag, args.seed)
    klo, khi = args.kfit

    starts = np.arange(0.0, args.t_end - args.width + 1e-9, args.step)
    print(f"seed {args.seed} ({args.tag}); fit slope over k in [{klo},{khi}] "
          f"(-5/3 = -1.67)\n")
    print("  window        n   E_tot slope  E_sol slope  E_dil slope  "
          "K_sol/K_tot  peak_k")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(starts)))
    rows = []
    for c, t0 in zip(colors, starts):
        win = (t0, t0 + args.width)
        try:
            k, Es, Ed, Et, n, (tlo, thi) = tavg(path, win)
        except SystemExit:
            continue
        s_t = fit_slope(k, Et, klo, khi)
        s_s = fit_slope(k, Es, klo, khi)
        s_d = fit_slope(k, Ed, klo, khi)
        m = k > 0
        Ksol = np.trapz(Es[m], k[m]); Ktot = np.trapz(Et[m], k[m])
        ipk = np.argmax(Et[1:]) + 1
        print(f"  [{tlo:.3f},{thi:.3f}]  {n:3d}  {s_t:9.2f}    {s_s:9.2f}    "
              f"{s_d:9.2f}    {Ksol/Ktot:.3f}      {k[ipk]:.1f}")
        rows.append((tlo, thi, n, s_t, s_s, s_d, Ksol / Ktot))

        mm = (k > 0) & (Et > 0)
        ax[0].loglog(k[mm], Et[mm] * k[mm] ** (5.0 / 3.0), lw=1.8, color=c,
                     label=f"[{tlo:.2f},{thi:.2f}]")
        kk, sl = local_slope(k, Et)
        ax[1].semilogx(kk, sl, lw=1.6, color=c)

    ax[0].set_xlabel("k"); ax[0].set_ylabel(r"$E_{tot}(k)\,k^{5/3}$")
    ax[0].set_title(r"compensated TOTAL spectrum (flat $\Rightarrow k^{-5/3}$)")
    ax[0].legend(frameon=False, fontsize=7); ax[0].grid(True, which="both", ls=":", alpha=0.4)
    ax[0].axvspan(klo, khi, color="gray", alpha=0.12)
    ax[1].axhline(-5.0 / 3.0, color="green", ls="--", lw=1.4, label="-5/3")
    ax[1].set_ylim(-4, 3); ax[1].set_xlabel("k")
    ax[1].set_ylabel(r"local slope $d\log E_{tot}/d\log k$")
    ax[1].set_title("TOTAL local slope"); ax[1].legend(frameon=False, fontsize=8)
    ax[1].grid(True, which="both", ls=":", alpha=0.4)
    ax[1].axvspan(klo, khi, color="gray", alpha=0.12)
    fig.suptitle(f"Total spectrum vs small averaging window  "
                 f"(seed {args.seed}, width {args.width})", y=1.0)
    fig.tight_layout()
    out = ROOT / "figs" / f"total_spectrum_windows_seed{args.seed}_w{args.width:.3f}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
