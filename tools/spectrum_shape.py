#!/usr/bin/env python3
"""Spectrum-shape diagnostics: separate spectral *energy* from spectral *cascade*.

A shell spectrum carries two physically distinct pieces of information that are
easy to conflate:

  * **Energy (bookkeeping / magnitude)** -- how much kinetic energy lives in a
    component: ``K = integral E(k) dk``, and its fractions
    (``K_dil/K_tot``, ``K_dil/K_sol``). This says nothing about turbulence; a
    single standing wave has dilatational energy.

  * **Turbulence (spectral shape / cascade)** -- does ``E(k)`` show a broadband
    power-law range (a cascade) or is it peaked / energy-containing? This is the
    ``d(log E)/d(log k)`` slope and the width of any power-law range, and is what
    distinguishes genuine turbulence from a few coherent modes.

This module reads the ``E_sol`` / ``E_dil`` / ``E_total`` shell spectra that the
solver already writes to ``<run>_spectra.h5`` (see
``solver/src/diagnostics/Spectra.cpp``) and reports the two axes separately. It
does **not** perform any new field decomposition -- it only re-presents the
spectra already emitted.

The slope/window helpers (``tavg``, ``local_slope``) are the canonical copies
shared by ``scripts/spectrum_components.py`` and ``scripts/total_spectrum_windows.py``.

CLI:
    python tools/spectrum_shape.py <run>_spectra.h5 [--t0 T0 --t1 T1]
        [--target-slope -5/3] [--tol 0.25]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np

COMPONENTS = ("E_total", "E_sol", "E_dil")


# --------------------------------------------------------------------------- io
def list_steps(path: Path) -> list[tuple[int, float]]:
    """Return sorted ``(step, time)`` for every ``step_NNNNNN`` group."""
    out = []
    with h5py.File(path, "r") as f:
        for name in f.keys():
            m = re.match(r"step_(\d+)", name)
            if not m:
                continue
            out.append((int(m.group(1)), float(f[name]["time"][0])))
    out.sort()
    return out


def load_step(path: Path, step: int) -> dict[str, np.ndarray]:
    """Load one step's ``k`` / ``E_total`` / ``E_sol`` / ``E_dil`` / ``time``."""
    with h5py.File(path, "r") as f:
        g = f[f"step_{step:06d}"]
        return {
            "k": np.asarray(g["k"], np.float64),
            "E_total": np.asarray(g["E_total"], np.float64),
            "E_sol": np.asarray(g["E_sol"], np.float64),
            "E_dil": np.asarray(g["E_dil"], np.float64),
            "time": float(g["time"][0]),
        }


def tavg(path, win):
    """Time-average the shell spectra over the window ``win = (t0, t1)``.

    Returns ``(k, E_sol, E_dil, E_total, n, (t_lo, t_hi))`` -- the signature
    relied on by spectrum_components / total_spectrum_windows.
    """
    k0 = None
    Es = Ed = Et = None
    n = 0
    tlo, thi = 1e9, -1e9
    with h5py.File(path, "r") as f:
        for key in f.keys():
            tt = float(f[key]["time"][0])
            if not (win[0] <= tt <= win[1]):
                continue
            g = f[key]
            k = np.asarray(g["k"], np.float64)
            if k0 is None:
                k0 = k
                Es = np.zeros_like(k)
                Ed = np.zeros_like(k)
                Et = np.zeros_like(k)
            Es += np.asarray(g["E_sol"])
            Ed += np.asarray(g["E_dil"])
            Et += np.asarray(g["E_total"])
            n += 1
            tlo, thi = min(tlo, tt), max(thi, tt)
    if n == 0:
        raise SystemExit(f"no spectra in window {win} of {path}")
    return k0, Es / n, Ed / n, Et / n, n, (tlo, thi)


# ----------------------------------------------------------- energy bookkeeping
def integrated_energy(k, E):
    """``integral E(k) dk`` over the resolved (k > 0) shells."""
    m = k > 0
    return float(np.trapz(E[m], k[m]))


def peak_k(k, E):
    """``(k_peak, E_peak)`` of the spectrum, excluding the k=0 mean bin."""
    if k.size < 2:
        return float("nan"), float("nan")
    ipk = int(np.argmax(E[1:])) + 1
    return float(k[ipk]), float(E[ipk])


# ------------------------------------------------------------ spectral shape
def local_slope(k, E, rel_floor=0.0):
    """``d(log E)/d(log k)`` on the strictly positive (k > 0, E > 0) shells.

    ``rel_floor``: if > 0, also drop shells where ``E < rel_floor * max(E)`` over
    the resolved range. This trims the roundoff tail below the dissipation
    roll-off, where ``E`` is ~0 and the slope is meaningless oscillation -- useful
    for plots. The default 0.0 keeps every positive shell, so callers that
    measure the slope (slope_at, powerlaw_range) are unaffected.

    Returns empty arrays when fewer than two shells qualify (e.g. an all-zero
    spectrum at t=0), since the gradient is undefined there.
    """
    m = (k > 0) & (E > 0)
    if rel_floor > 0 and m.any():
        m &= E > rel_floor * E[m].max()
    if int(m.sum()) < 2:
        return k[m], np.zeros(int(m.sum()))
    lk = np.log(k[m])
    lE = np.log(E[m])
    return k[m], np.gradient(lE, lk)


def smooth_slope(k, E, dlnk=0.4, rel_floor=0.0):
    """Local log-log slope via least-squares fit over a +/- ``dlnk`` ln-k window.

    For *display*: the raw 2-point ``local_slope`` is noisy because the shell
    bins are linear in k (hundreds of densely-packed bins at high k), so adjacent
    differences swing wildly. Fitting log E vs log k over a fixed ln-k window
    averages many bins where they are dense (high k) and few where they are
    sparse (low k), giving a readable curve without distorting the trend. The
    measurement path (``slope_at`` / ``powerlaw_range``) keeps using the raw
    2-point slope, so reported numbers are unaffected.
    """
    m = (k > 0) & (E > 0)
    if rel_floor > 0 and m.any():
        m &= E > rel_floor * E[m].max()
    kk = k[m]
    if kk.size < 3:
        return kk, np.zeros(kk.size)
    lk = np.log(kk)
    lE = np.log(E[m])
    sl = np.empty(kk.size)
    for i in range(kk.size):
        w = np.abs(lk - lk[i]) <= dlnk
        if int(w.sum()) < 3:                       # widen to nearest 3 points
            w = np.zeros(kk.size, bool)
            w[np.argsort(np.abs(lk - lk[i]))[:3]] = True
        sl[i] = np.polyfit(lk[w], lE[w], 1)[0]
    return kk, sl


def slope_at(k, E, kq):
    """Local slope interpolated at wavenumber ``kq`` (NaN if unresolved)."""
    kk, sl = local_slope(k, E)
    if kk.size < 2 or kq < kk.min() or kq > kk.max():
        return float("nan")
    return float(np.interp(kq, kk, sl))


def powerlaw_range(k, E, target, tol):
    """Longest contiguous k-range whose local slope is within ``tol`` of ``target``.

    Returns ``dict(k_lo, k_hi, octaves, nbins)``; all-zero / NaN endpoints when
    no qualifying range exists. ``octaves = log2(k_hi / k_lo)`` is the cascade
    width -- the headline "is there a turbulent range, and how wide" number.
    """
    kk, sl = local_slope(k, E)
    near = np.abs(sl - target) < tol
    best_i = best_len = 0
    i = 0
    while i < near.size:
        if not near[i]:
            i += 1
            continue
        j = i
        while j < near.size and near[j]:
            j += 1
        if j - i > best_len:
            best_len, best_i = j - i, i
        i = j
    if best_len == 0:
        return {"k_lo": float("nan"), "k_hi": float("nan"),
                "octaves": 0.0, "nbins": 0}
    k_lo = float(kk[best_i])
    k_hi = float(kk[best_i + best_len - 1])
    octaves = float(np.log2(k_hi / k_lo)) if k_lo > 0 else 0.0
    return {"k_lo": k_lo, "k_hi": k_hi, "octaves": octaves, "nbins": int(best_len)}


def classify_shape(k, E, target, tol, min_octaves=1.0):
    """``"cascade"`` if a power-law range >= ``min_octaves`` exists, else peaked.

    A spectrum dominated by its peak with no broadband power law is
    ``"energy-containing/peaked"`` -- it carries energy but no cascade, i.e.
    spectral energy without spectral turbulence.
    """
    rng = powerlaw_range(k, E, target, tol)
    return "cascade" if rng["octaves"] >= min_octaves else "energy-containing/peaked"


# --------------------------------------------------------------- slope parsing
def parse_slope(text: str) -> float:
    """Parse a reference slope like ``-5/3`` or ``-2`` into a float."""
    text = text.strip()
    if "/" in text:
        num, den = text.split("/")
        return float(num) / float(den)
    return float(text)


# ------------------------------------------------------------------------- cli
def report(path: Path, win, target, tol):
    """Print the energy and turbulence blocks for a spectra file + window."""
    k, Es, Ed, Et, n, (tlo, thi) = tavg(path, win)
    spec = {"E_total": Et, "E_sol": Es, "E_dil": Ed}

    print(f"{path.name}")
    print(f"  window {win} -> {n} spectra (actual t in [{tlo:.4f},{thi:.4f}])\n")

    # -- energy: magnitude / bookkeeping only --------------------------------
    Ktot = integrated_energy(k, Et)
    Ksol = integrated_energy(k, Es)
    Kdil = integrated_energy(k, Ed)
    print("  === dilatational ENERGY (bookkeeping: integrated magnitude) ===")
    print(f"    K_total = {Ktot:.4e}")
    print(f"    K_sol   = {Ksol:.4e}   (K_sol/K_tot = {Ksol/Ktot:.4f})")
    print(f"    K_dil   = {Kdil:.4e}   (K_dil/K_tot = {Kdil/Ktot:.4f}, "
          f"K_dil/K_sol = {Kdil/max(Ksol,1e-30):.4f})")
    pk = {c: peak_k(k, spec[c])[0] for c in COMPONENTS}
    print(f"    peak-k:  total={pk['E_total']:.2f}  sol={pk['E_sol']:.2f}  "
          f"dil={pk['E_dil']:.2f}\n")

    # -- turbulence: spectral shape / cascade only ---------------------------
    print(f"  === dilatational TURBULENCE (spectral shape: cascade character) ===")
    print(f"    target slope = {target:+.2f}  (tol {tol})")
    print(f"    {'component':9s} {'slope@k6':>9s} {'slope@k16':>10s}   "
          f"{'power-law range':<28s} verdict")
    for c in COMPONENTS:
        E = spec[c]
        rng = powerlaw_range(k, E, target, tol)
        if rng["nbins"]:
            rng_s = (f"k={rng['k_lo']:.1f}-{rng['k_hi']:.1f} "
                     f"({rng['octaves']:.1f} oct, {rng['nbins']} bins)")
        else:
            rng_s = "none"
        verdict = classify_shape(k, E, target, tol)
        print(f"    {c:9s} {slope_at(k,E,6):+9.2f} {slope_at(k,E,16):+10.2f}   "
              f"{rng_s:<28s} {verdict}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spectra", type=Path, help="<run>_spectra.h5 file")
    ap.add_argument("--t0", type=float, default=-1e9, help="window start time")
    ap.add_argument("--t1", type=float, default=1e9, help="window end time")
    ap.add_argument("--target-slope", default="-5/3",
                    help="reference cascade slope (e.g. -5/3, -2). Default -5/3.")
    ap.add_argument("--tol", type=float, default=0.25,
                    help="slope tolerance for the power-law range scan")
    args = ap.parse_args()

    report(args.spectra, (args.t0, args.t1),
           parse_slope(args.target_slope), args.tol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
