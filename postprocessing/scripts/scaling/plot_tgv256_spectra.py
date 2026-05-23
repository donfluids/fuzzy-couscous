#!/usr/bin/env python3
"""TGV Re=1600 256^3 spectra: snapshots at the key times of the decay, the
local spectral slope, the grid-convergence comparison against 128^3 / 64^3
at the dissipation peak, and the TKE / eps history.

The 256^3 run covers t in [0, 12], including the laminar-vortex phase
(t < 4), vortex breakdown (t ~ 4-7), the dissipation peak (t ~ 8-9 for
Re=1600), and decay (t > 9). At 256^3 the inertial range should be wide
enough (k ~ 4..30) to show a clean -5/3 plateau around the peak.
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
    runs: dict[str, dict] = {
        "256³": {
            "spec":  run_dir("out_tgv256_hyper2/tgv_re1600_256_hyper2_spectra.h5"),
            "stats": run_dir("out_tgv256_hyper2/tgv_re1600_256_hyper2_stats.csv"),
            "nyq":   128,
        },
        "128³": {
            "spec":  run_dir("out_tgv128_hyper2/tgv_re1600_128_hyper2_spectra.h5"),
            "stats": run_dir("out_tgv128_hyper2/tgv_re1600_128_hyper2_stats.csv"),
            "nyq":   64,
        },
        "64³": {
            "spec":  run_dir("out_tgv64_hyper2/tgv_re1600_64_hyper2_spectra.h5"),
            "stats": run_dir("out_tgv64_hyper2/tgv_re1600_64_hyper2_stats.csv"),
            "nyq":   32,
        },
    }
    if not runs["256³"]["spec"].exists():
        print(f"missing 256^3 spectra: {runs['256³']['spec']}", file=sys.stderr)
        return 1
    for name in list(runs):
        if not runs[name]["spec"].exists():
            print(f"  ({name}: not present, skipping overlay)")
            del runs[name]

    s256 = list_steps(runs["256³"]["spec"])
    t_max = s256[-1][1]
    snapshot_times = sorted({round(t, 2) for t in (0.5, 2.0, 5.0, 8.0, 9.0, t_max)
                             if t <= t_max + 1e-6})

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    cmap = plt.get_cmap("plasma")

    # ----- Panel (0,0): 256^3 spectra at key times -----
    ax = axes[0, 0]
    for i, tt in enumerate(snapshot_times):
        step, _ = pick_nearest(s256, tt)
        d = load(runs["256³"]["spec"], step)
        color = cmap(i / max(1, len(snapshot_times) - 1))
        ax.loglog(d["k"][1:], d["E_total"][1:], "-", color=color, lw=1.7,
                  label=f"t={d['time']:.2f}")

    # k^-5/3 reference anchored at k=8 on the t~peak curve
    peak_target = min(snapshot_times, key=lambda t: abs(t - 8.5))
    step_peak, _ = pick_nearest(s256, peak_target)
    dpeak = load(runs["256³"]["spec"], step_peak)
    k_arr, E_arr = dpeak["k"][1:], dpeak["E_total"][1:]
    idx = int(np.argmin(np.abs(k_arr - 8.0)))
    k_ref = np.geomspace(3.0, 40.0, 32)
    ref = E_arr[idx] * (k_ref / k_arr[idx]) ** (-5.0 / 3.0)
    ax.loglog(k_ref, ref, "k:", lw=1.5, label=r"$\propto k^{-5/3}$")

    ax.axvline(runs["256³"]["nyq"], color="grey", ls="-.", alpha=0.6,
               label="256³ Nyquist k=128")
    ax.set_xlim(1, 200)
    ax.set_ylim(1e-15, 0.2)
    ax.set_xlabel("wavenumber k")
    ax.set_ylabel("E(k)")
    ax.set_title("TGV Re=1600, 256³ spectra")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, ncol=2)

    # ----- Panel (0,1): local spectral slope on 256^3 -----
    ax = axes[0, 1]
    for i, tt in enumerate(snapshot_times):
        step, _ = pick_nearest(s256, tt)
        d = load(runs["256³"]["spec"], step)
        k, E = d["k"][1:], d["E_total"][1:]
        good = (E > 0) & (k > 0)
        kl, El = np.log(k[good]), np.log(E[good])
        slope = np.gradient(El, kl)
        ax.semilogx(k[good], slope, "-",
                    color=cmap(i / max(1, len(snapshot_times) - 1)),
                    label=f"t={d['time']:.2f}")
    ax.axhline(-5/3, color="k", linestyle="--", lw=1, label=r"$-5/3$")
    ax.set_xlim(1, 200)
    ax.set_ylim(-6, 1)
    ax.set_xlabel("wavenumber k")
    ax.set_ylabel(r"local slope $d\log E / d\log k$")
    ax.set_title("Local spectral slope (256³)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    # ----- Panel (1,0): grid convergence at peak dissipation time -----
    ax = axes[1, 0]
    peak_time = peak_target
    colors = {"256³": "C3", "128³": "C0", "64³": "C2"}
    for name in ("256³", "128³", "64³"):
        if name not in runs:
            continue
        steps = list_steps(runs[name]["spec"])
        step, _ = pick_nearest(steps, peak_time)
        d = load(runs[name]["spec"], step)
        ax.loglog(d["k"][1:], d["E_total"][1:], "-",
                  color=colors[name], lw=1.6,
                  label=f"{name} t={d['time']:.2f}")
        ax.axvline(runs[name]["nyq"], color=colors[name], ls=":", alpha=0.45)
    k_ref = np.geomspace(3.0, 40.0, 32)
    ref = E_arr[idx] * (k_ref / k_arr[idx]) ** (-5.0 / 3.0)
    ax.loglog(k_ref, ref, "k:", lw=1.5, label=r"$\propto k^{-5/3}$")
    ax.set_xlim(1, 200)
    ax.set_ylim(1e-15, 0.2)
    ax.set_xlabel("wavenumber k")
    ax.set_ylabel("E(k)")
    ax.set_title(f"Grid convergence at t ≈ {peak_time:.1f} (peak dissipation)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    # ----- Panel (1,1): TKE + eps_total history, all grids -----
    ax = axes[1, 1]
    ax2 = ax.twinx()
    styles = {"256³": ("-", 1.6), "128³": ("--", 1.2), "64³": (":", 1.0)}
    for name, info in runs.items():
        if not info["stats"].exists():
            continue
        df = pd.read_csv(info["stats"])
        ls, lw = styles[name]
        ax.plot(df["time"], df["eps_total"], color="C3", ls=ls, lw=lw,
                label=f"{name} $\\varepsilon_{{tot}}$")
        ax2.plot(df["time"], df["tke"], color="C0", ls=ls, lw=lw,
                 label=f"{name} tke")
    ax.set_xlim(0, max(t_max, 4.0))
    ax.set_xlabel("time")
    ax.set_ylabel("dissipation rate", color="C3")
    ax.tick_params(axis="y", labelcolor="C3")
    ax2.set_ylabel("tke", color="C0")
    ax2.tick_params(axis="y", labelcolor="C0")
    for tt in snapshot_times:
        ax.axvline(tt, color="gray", alpha=0.18, linestyle=":")
    ax.set_title("Dissipation and TKE history (all grids)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"TGV Re=1600, 256³ — full decay through t={t_max:.2f}",
        y=1.00,
    )
    fig.tight_layout()
    out = HERE / "paper/figures/tgv256_spectra_full.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")

    # Print quick numerical summary
    print()
    df = pd.read_csv(runs["256³"]["stats"])
    peak_idx = df["eps_total"].idxmax()
    print(f"Run summary (256³):")
    print(f"  t range          : [{df['time'].min():.3f}, {df['time'].max():.3f}]")
    print(f"  dissipation peak : t = {df.loc[peak_idx, 'time']:.3f}, eps_total = {df.loc[peak_idx, 'eps_total']:.4e}")
    print(f"  tke decay        : {df.iloc[0]['tke']:.4e} -> {df.iloc[-1]['tke']:.4e} ({df.iloc[-1]['tke']/df.iloc[0]['tke']*100:.1f}% remaining)")
    # Inertial-range slope at peak, k in [4, 16]
    with h5py.File(runs["256³"]["spec"], "r") as f:
        keys = sorted([k for k in f.keys() if re.match(r"step_\d+", k)])
        times = [(k, float(f[k]["time"][0])) for k in keys]
        nearest_step, _ = min(times, key=lambda kt: abs(kt[1] - df.loc[peak_idx, "time"]))
        g = f[nearest_step]
        k, E = g["k"][1:], g["E_total"][1:]
        good = (E > 0)
        sl = np.gradient(np.log(E[good]), np.log(k[good]))
        inertial = (k[good] >= 4) & (k[good] <= 16)
        print(f"  local slope k∈[4,16] @ peak: mean = {sl[inertial].mean():.3f}, ref = -1.667")

    return 0


if __name__ == "__main__":
    sys.exit(main())
