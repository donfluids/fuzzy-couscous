#!/usr/bin/env python3
"""TGV Re=1600 128^3 spectra: snapshots at key times + local slope + KE/eps
history. Overlays existing 64^3 and (truncated) 256^3 runs when their
HDF5 outputs are present, otherwise plots 128^3 alone.

The 128^3 run goes to t=12, which covers the full Re=1600 dissipation
history: the laminar-vortex phase (t<4), vortex breakdown (t~4-7), the
dissipation peak (t~8-9 for Re=1600), and decay (t>9). The peak-time
spectrum should show a clean inertial range over k~3..15 ish.
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

HERE = Path(__file__).resolve().parent.parent  # repo root


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
        "128³": {
            "spec":  HERE / "out_tgv128_hyper2/tgv_re1600_128_hyper2_spectra.h5",
            "stats": HERE / "out_tgv128_hyper2/tgv_re1600_128_hyper2_stats.csv",
            "style": {"color": "C0", "ls": "-"},
            "nyq":   64,
        },
        "256³": {
            "spec":  HERE / "out_tgv256_hyper2/tgv_re1600_256_hyper2_spectra.h5",
            "stats": HERE / "out_tgv256_hyper2/tgv_re1600_256_hyper2_stats.csv",
            "style": {"color": "C3", "ls": "-"},
            "nyq":   128,
        },
        "64³": {
            "spec":  HERE / "out_tgv64_hyper2/tgv_re1600_64_hyper2_spectra.h5",
            "stats": HERE / "out_tgv64_hyper2/tgv_re1600_64_hyper2_stats.csv",
            "style": {"color": "C2", "ls": "--"},
            "nyq":   32,
        },
    }
    if not runs["128³"]["spec"].exists():
        print(f"missing 128^3 spectra: {runs['128³']['spec']}", file=sys.stderr)
        return 1
    for name in list(runs):
        if not runs[name]["spec"].exists():
            print(f"  ({name}: not present, skipping overlay)")
            del runs[name]

    s128 = list_steps(runs["128³"]["spec"])
    t_max = s128[-1][1]
    target_times = sorted({round(t, 2) for t in (0.5, 2.0, 5.0, 8.0, 9.0, t_max)
                           if t <= t_max + 1e-6})

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    cmap = plt.get_cmap("plasma")

    # Panel 0: 128^3 spectra at multiple times; overlay 64^3/256^3 at last time
    for i, tt in enumerate(target_times):
        step, _ = pick_nearest(s128, tt)
        d = load(runs["128³"]["spec"], step)
        color = cmap(i / max(1, len(target_times) - 1))
        axes[0].loglog(d["k"][1:], d["E_total"][1:],
                       "-", color=color, lw=1.7,
                       label=f"128³ t={d['time']:.2f}")
    # overlay other grids at t≈t_max for cross-resolution comparison
    for name in ("64³", "256³"):
        if name not in runs:
            continue
        steps = list_steps(runs[name]["spec"])
        step, _ = pick_nearest(steps, t_max)
        d = load(runs[name]["spec"], step)
        axes[0].loglog(d["k"][1:], d["E_total"][1:],
                       ls=runs[name]["style"]["ls"],
                       color="k" if name == "64³" else "0.45",
                       lw=1.2,
                       label=f"{name} t={d['time']:.2f}")

    # k^-5/3 reference anchored at k=6 on 128^3 t=t_max
    step_last, _ = pick_nearest(s128, t_max)
    dlast = load(runs["128³"]["spec"], step_last)
    k_arr, E_arr = dlast["k"][1:], dlast["E_total"][1:]
    idx = int(np.argmin(np.abs(k_arr - 6.0)))
    k_ref = np.geomspace(3.0, 20.0, 32)
    ref = E_arr[idx] * (k_ref / k_arr[idx]) ** (-5.0 / 3.0)
    axes[0].loglog(k_ref, ref, "k:", lw=1.5, label=r"$\propto k^{-5/3}$")

    axes[0].axvline(runs["128³"]["nyq"], color="grey", ls="-.", alpha=0.6,
                    label="128³ Nyquist k=64")
    axes[0].set_xlim(1, 200)
    axes[0].set_ylim(1e-13, 0.2)
    axes[0].set_xlabel("wavenumber k")
    axes[0].set_ylabel("E(k)")
    axes[0].set_title("TGV Re=1600, 128³ spectra (+ overlay at t=t_max)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="lower left", fontsize=7.5, ncol=2)

    # Panel 1: local slope at the dissipation-peak times on 128^3
    slope_times = [t for t in (2.0, 5.0, 8.0, 9.0, t_max) if t <= t_max + 1e-6]
    for i, tt in enumerate(slope_times):
        step, _ = pick_nearest(s128, tt)
        d = load(runs["128³"]["spec"], step)
        k, E = d["k"][1:], d["E_total"][1:]
        good = (E > 0) & (k > 0)
        kl, El = np.log(k[good]), np.log(E[good])
        slope = np.gradient(El, kl)
        axes[1].semilogx(k[good], slope, "-",
                         color=cmap(i / max(1, len(slope_times) - 1)),
                         label=f"t={d['time']:.2f}")
    axes[1].axhline(-5/3, color="k", linestyle="--", lw=1,
                    label=r"$-5/3$")
    axes[1].set_xlim(1, 200)
    axes[1].set_ylim(-6, 1)
    axes[1].set_xlabel("wavenumber k")
    axes[1].set_ylabel(r"local slope $d\log E / d\log k$")
    axes[1].set_title("Local spectral slope (128³)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower left")

    # Panel 2: TKE / eps_total history; show all available grids
    ax = axes[2]; ax2 = ax.twinx()
    for name, info in runs.items():
        if not info["stats"].exists():
            continue
        df = pd.read_csv(info["stats"])
        st = info["style"]
        ax.plot(df["time"], df["eps_total"],
                color="C3", ls=st["ls"], lw=1.4 if name == "128³" else 0.9,
                label=f"{name} $\\varepsilon_{{tot}}$")
        ax2.plot(df["time"], df["tke"],
                 color="C0", ls=st["ls"], lw=1.4 if name == "128³" else 0.9,
                 label=f"{name} tke")
    ax.set_xlim(0, max(t_max, 4.0))
    ax.set_xlabel("time")
    ax.set_ylabel("dissipation rate", color="C3")
    ax.tick_params(axis="y", labelcolor="C3")
    ax2.set_ylabel("tke", color="C0")
    ax2.tick_params(axis="y", labelcolor="C0")
    for tt in target_times:
        ax.axvline(tt, color="gray", alpha=0.2, linestyle=":")
    ax.set_title("Dissipation and TKE history")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=7.5)
    ax2.legend(loc="upper right", fontsize=7.5)

    fig.suptitle(
        f"TGV Re=1600, 128³ — full decay through t={t_max:.2f}",
        y=1.02,
    )
    fig.tight_layout()
    out = HERE / "paper/figures/tgv128_spectra.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
