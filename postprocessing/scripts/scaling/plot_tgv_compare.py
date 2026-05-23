#!/usr/bin/env python3
"""Overlay TGV-Re1600 spectra from 64^3 and 256^3 runs at matched times,
plus a local-slope diagnostic and the dissipation history.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_PP = next(p for p in _Path(__file__).resolve().parents if (p / "paths.py").is_file())
for _d in (_PP, _PP / "tools", _PP / "scripts"):
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
from paths import REPO_ROOT as HERE, RUNS, DATA, FIGS, run_dir  # noqa: E402


def list_steps(path: Path) -> list[tuple[int, float]]:
    with h5py.File(path, "r") as f:
        out = []
        for name in f.keys():
            m = re.match(r"step_(\d+)", name)
            if not m:
                continue
            out.append((int(m.group(1)), float(f[name]["time"][0])))
    out.sort()
    return out


def load(path: Path, step: int) -> dict:
    with h5py.File(path, "r") as f:
        g = f[f"step_{step:06d}"]
        return {
            "k":       g["k"][...],
            "E_total": g["E_total"][...],
            "E_sol":   g["E_sol"][...],
            "E_dil":   g["E_dil"][...],
            "time":    float(g["time"][0]),
        }


def pick_nearest(steps_times: list, target: float):
    return min(steps_times, key=lambda st: abs(st[1] - target))


def main() -> int:
    spec_256 = run_dir("out_tgv256_hyper2/tgv_re1600_256_hyper2_spectra.h5")
    spec_64  = run_dir("out_tgv64_hyper2/tgv_re1600_64_hyper2_spectra.h5")
    stats_256 = run_dir("out_tgv256_hyper2/tgv_re1600_256_hyper2_stats.csv")
    stats_64  = run_dir("out_tgv64_hyper2/tgv_re1600_64_hyper2_stats.csv")
    for p in (spec_256, spec_64, stats_256, stats_64):
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            return 1

    s256 = list_steps(spec_256)
    s64  = list_steps(spec_64)
    t_max_256 = s256[-1][1]
    target_times = [0.5, 1.0, 2.0, 3.0, t_max_256]
    target_times = sorted(set(round(t, 2) for t in target_times if t <= t_max_256))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    cmap = plt.get_cmap("plasma")

    # Panel 0: spectra overlay at multiple times (256 = solid, 64 = dashed)
    for i, tt in enumerate(target_times):
        step_a, t_a = pick_nearest(s256, tt)
        step_b, t_b = pick_nearest(s64,  tt)
        d_a = load(spec_256, step_a)
        d_b = load(spec_64,  step_b)
        color = cmap(i / max(1, len(target_times) - 1))
        k_a, E_a = d_a["k"][1:], d_a["E_total"][1:]
        k_b, E_b = d_b["k"][1:], d_b["E_total"][1:]
        axes[0].loglog(k_a, E_a, "-",  color=color, lw=1.6,
                       label=f"256³, t={t_a:.2f}")
        axes[0].loglog(k_b, E_b, "--", color=color, lw=1.2,
                       label=f"64³,  t={t_b:.2f}")

    # Anchor a -5/3 reference to the 256 t=last curve in the k~4..8 window.
    step_last, _ = pick_nearest(s256, target_times[-1])
    dlast = load(spec_256, step_last)
    k_arr = dlast["k"][1:]
    E_arr = dlast["E_total"][1:]
    idx = np.argmin(np.abs(k_arr - 6.0))
    k_ref = np.geomspace(3.0, 18.0, 32)
    ref = E_arr[idx] * (k_ref / k_arr[idx]) ** (-5.0 / 3.0)
    axes[0].loglog(k_ref, ref, "k:", lw=1.5, label=r"$\propto k^{-5/3}$ (ref)")

    axes[0].axvline(32, color="grey", ls="-.", alpha=0.6,
                    label="64³ Nyquist k=32")
    axes[0].set_xlim(1, 200)
    axes[0].set_ylim(1e-13, 0.2)
    axes[0].set_xlabel("wavenumber k")
    axes[0].set_ylabel("E(k)")
    axes[0].set_title("TGV Re=1600 spectra: 256³ vs 64³ at matched times")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="lower left", fontsize=7.5, ncol=2)

    # Panel 1: local spectral slope at the latest 256 time
    for i, tt in enumerate([1.0, 2.0, 3.0, target_times[-1]]):
        step, _ = pick_nearest(s256, tt)
        d = load(spec_256, step)
        k = d["k"][1:]
        E = d["E_total"][1:]
        good = (E > 0) & (k > 0)
        kl = np.log(k[good]); El = np.log(E[good])
        slope = np.gradient(El, kl)
        axes[1].semilogx(k[good], slope, "-",
                         color=cmap(i / 3),
                         label=f"256³, t={d['time']:.2f}")
    axes[1].axhline(-5/3, color="k", linestyle="--", lw=1,
                    label=r"$-5/3$")
    axes[1].set_xlim(1, 200)
    axes[1].set_ylim(-5, 1)
    axes[1].set_xlabel("wavenumber k")
    axes[1].set_ylabel(r"local slope $d\log E / d\log k$")
    axes[1].set_title("Local spectral slope (256³)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower left")

    # Panel 2: KE / dissipation history overlay
    df256 = pd.read_csv(stats_256)
    df64  = pd.read_csv(stats_64)
    ax = axes[2]
    ax.plot(df256["time"], df256["eps_total"],
            color="C3", label="256³ $\\varepsilon_{tot}$")
    ax.plot(df64["time"], df64["eps_total"],
            color="C3", linestyle="--", label="64³ $\\varepsilon_{tot}$")
    ax.set_xlim(0, max(df256["time"].max(), 4.0))
    ax.set_xlabel("time")
    ax.set_ylabel("dissipation rate", color="C3")
    ax.tick_params(axis="y", labelcolor="C3")
    ax2 = ax.twinx()
    ax2.plot(df256["time"], df256["tke"], color="C0", label="256³ tke")
    ax2.plot(df64["time"], df64["tke"], color="C0", linestyle="--",
             label="64³ tke")
    ax2.set_ylabel("tke", color="C0")
    ax2.tick_params(axis="y", labelcolor="C0")
    for tt in target_times:
        ax.axvline(tt, color="gray", alpha=0.2, linestyle=":")
    ax.set_title("Dissipation and TKE history")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        f"TGV Re=1600, 256³ vs 64³ — 256³ run stopped at t={t_max_256:.2f} "
        "(cascade forming, not yet at the t≈8 -5/3 plateau)",
        y=1.02,
    )
    fig.tight_layout()
    out = HERE / "paper/figures/tgv256_vs_64_spectra.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
