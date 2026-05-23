# Roadmap — future & deferred work

Known future work, grouped by the [`STRATEGY.md`](../STRATEGY.md) tracks. This is
the single place "all futures are noted"; each item carries enough context and a
code/test pointer to be picked up later. Dates are absolute.

## Core solver fidelity

- **DCT spectra wavenumber convention (open).** The DCT shell-binning convention
  is unresolved: bins on `π/L` (N shells) vs. an integer `2π/L` grid
  (`N/2` = Nyquist). The cube-corner fix is done (`ef4fe43`); the `π/L → 2π/L`
  normalization is still pending. With `L = 2π` integer `k` follows directly.
  Affects `diagnostics/Spectra.cpp` and `postprocessing/tools/spectrum_shape.py`.
- **Compact-scheme coverage.** The 10th-order conservative compact flux
  reconstruction landed serial-only (`904d98d`); an MPI port + verification
  parity is future work (`numerics/CompactScheme.cpp`).

## Physics scope extension (multi-fluid → BHR)

- **JWL afterburning (deferred).** Reactive heat release for the products EOS is
  a deferred follow-on; the current JWL branch is non-reactive
  (`physics/JWL.hpp`, `physics/MixtureEOS.hpp`).
- **Multifluid double-flux non-conservation (study, open).** The two-γ
  double-flux scheme gains energy ~1 % over a few steps for strong contrasts
  (pre-existing; not the MPI port nor the floor). The conservative-flux option
  (`conservative = true`) trades contact oscillations for energy conservation;
  characterizing the trade-off across Atwood numbers is open
  (`physics/Multifluid.cpp`; G-boundedness metric already reported).
- **BHR two-way coupling validation.** The feedback path (Reynolds-stress
  divergence + dissipative heating, `f_k`-blended) needs broader validation
  against the LES `a/b` budget beyond the 1-D/seed tests
  (`turbulence/BHR.cpp`, `postprocessing/scripts/bhr_*`).

## Verification & validation

- **Experimental-data cross-check (pending data).** The manuscript's reference
  experimental measurements are TBD; wiring the comparison and reporting the
  discrepancy is future work once the dataset matures (see STRATEGY metrics).
- **Multi-node MPI scaling for 768³ ensembles.** Single-node strong scaling is
  verified; the production target is multi-node 768³. Drivers in `scaling/` write
  to `runs/scaling/`; large-scale runs + a scaling report are future work.

## Diagnostics & community delivery

- **Reorg follow-ups (from the 2026-05-23 reorganization):**
  - Prune the merged branches `abv-artificial-diffusivity` and
    `spectra-energy-vs-turbulence` once the reorg stack lands.
  - Consider folding `data/` (derived `.npz`) into a `runs/_derived/` location;
    kept at repo root for now to limit blast radius.
  - Dedupe the redundant `import sys` / `from pathlib import Path` left in some
    rewired `postprocessing/` scripts (cosmetic; a lint pass).
  - Run `/ce-compound` to capture the reorg learnings (path-anchoring pattern,
    `runs/` taxonomy, `out_dir` convention, LaTeX/PDF build) — `docs/solutions/`
    is currently sparse.
- **Packaging.** Consider promoting `postprocessing/` to an installable package
  (`pyproject.toml`) so imports work without the `sys.path` bootstrap and scripts
  gain console entry points.
