---
title: "Repository conventions after the 2026-05-23 reorg: where to run and how things are organized"
date: 2026-05-23
category: conventions
module: repo-organization
problem_type: convention
component: development_workflow
severity: high
applies_when:
  - "Starting a new solver run or parameter sweep"
  - "Adding or editing post-processing / analysis scripts"
  - "Building the solver (serial or MPI)"
  - "Looking for run outputs, figures, or derived data"
tags: [runs-layout, postprocessing, paths-resolver, out-dir-convention, build-dirs, repo-organization]
---

# Repository conventions after the 2026-05-23 reorg: where to run and how things are organized

## Context

On 2026-05-23 the repository was reorganized (see
`docs/plans/2026-05-23-001-refactor-repo-reorganization-plan.md`, status
`completed`). Before the reorg, run outputs were scattered across four
locations, 78 of 90 configs wrote a bare `out_dir` into the current working
directory, and analysis scripts split between `RUNS = ROOT/"runs"` and
`RUNROOT = ROOT/"solver"`. A future agent (or the author, months later) needs to
know the single set of conventions that now hold, so new runs land in the right
place and the analysis pipeline stays connected. This doc is the quick reference;
the deep reference lives in [`docs/`](../../README.md).

## Guidance

**Always run the solver from the repository root.** TOML configs set
`out_dir = "runs/<category>/<name>"` (relative to the CWD), and the solver
creates the nested path via `std::filesystem::create_directories`
(`solver/src/io/HDF5Writer.cpp`). Running from anywhere else scatters outputs
again.

### Where a new run goes

All run output lives under `runs/<category>/out_*/`. Categories are defined once
in `postprocessing/paths.py` (`category_of` / `run_dir`):

| Category | Holds | Rule (run-dir name) |
|----------|-------|------|
| `blast/`   | single- and multi-fluid blast runs | starts `out_blast` |
| `tgv/`     | Taylor-Green vortex | starts `out_tgv` |
| `chamber/` | closed-chamber set (sf/tg/tnt + paper Case 1) | contains `_chamber`, or `out_paper` |
| `tnt/`     | TNT/JWL free-air & smoke | starts `out_tnt` (non-chamber) |
| `bhr/`     | BHR / RANS model runs | starts `out_bhr` or `out_rans` |
| `scaling/` | strong-scaling sweeps, PGO, logs | scaling drivers + `out_scaling` |
| `hit/`     | forced homogeneous isotropic turbulence | starts `out_hit` |
| `misc/`    | anything matching no rule | fallback (never dropped) |

To add a new run: copy an existing `solver/examples/*.toml`, set
`out_dir = "runs/<category>/out_<name>"` (matching the taxonomy), and run from
the repo root. If a new case family is introduced, add a rule to
`paths.py::_RULES` and a `runs/<category>/.gitkeep`; that one edit teaches every
script and the migration tool about it. See [`runs/README.md`](../../../runs/README.md).

### How to build

Run cmake from the **repo root** (not from `solver/`):

```bash
# Serial (OpenMP)
cmake -S solver -B solver/build -DCMAKE_BUILD_TYPE=Release
cmake --build solver/build -j

# MPI (uses the ABI-safe wrappers; see Related)
./solver/cmake/configure-mpi.sh            # configures solver/build_mpi
cmake --build solver/build_mpi -j
./solver/run_mpi.sh ctest --test-dir solver/build_mpi --output-on-failure
```

Build dirs are `solver/build` (serial) and `solver/build_mpi` (MPI) — both under
`solver/`, both gitignored. Do **not** build into a repo-root `build_mpi/` (an
old `configure-mpi.sh` did this; it was fixed because the scaling drivers expect
`solver/build_mpi/blast_les_mpi`).

### How to run a config

```bash
# from the repo root
./solver/build/blast_les solver/examples/chamber_smoke.toml
./solver/run_mpi.sh mpirun -n 4 ./solver/build_mpi/blast_les_mpi solver/examples/paper_case1.toml
```

### Where analysis lives and how it finds runs

All post-processing is under `postprocessing/`:
`tools/` (reusable, path passed on the CLI), `scripts/` (case-specific, resolve
runs by name), `scripts/scaling/` (sweep analysis). Scripts import the shared
resolver and never hardcode a category:

```python
from paths import REPO_ROOT, RUNS, DATA, FIGS, run_dir
rd = run_dir("out_blast_128_budget_seed1")   # -> runs/blast/out_blast_128_budget_seed1
```

Figures are written to `figs/`; derived `.npz` products to `data/` (both
gitignored). Run the analysis tests with
`python3 -m unittest discover -s postprocessing/tests`. See
[`postprocessing/README.md`](../../../postprocessing/README.md).

### Moving / consolidating runs

`postprocessing/scripts/migrate_runs.py` is the reversible consolidation tool
(`--dry-run` default / `--apply` / `--rollback`); every move is recorded in
`runs/MANIFEST.md`. It fails fast on a destination collision (two source dirs
sharing a name) rather than overwriting — resolve by keeping one and renaming the
other before re-running.

## Why This Matters

The whole point of the reorg was a *connected* tree: one place runs live, one
resolver every script trusts, one `out_dir` convention so a config re-runs into
the same place the analysis reads from. Violating any of these silently
regresses it — e.g., running from `solver/` recreates the bare-CWD scatter, or
hardcoding `runs/out_x` (instead of `run_dir`) reads an empty path because runs
now sit under `runs/<category>/`. Run data (179 GB) is gitignored and on one
filesystem, so consolidation moves are instant and reversible — but only if the
manifest discipline is kept.

## When to Apply

- Before starting any solver run or sweep — confirm CWD is the repo root and
  `out_dir` follows `runs/<category>/`.
- When writing a new analysis script — import from `paths`, use `run_dir(name)`,
  write figures to `FIGS`.
- When building — use `solver/build` / `solver/build_mpi`, not repo-root or
  ad-hoc dirs.

## Examples

Stale-build-cache gotcha (cross-machine): a build dir created at a different
absolute path carries a CMake cache that refuses to reconfigure ("source ...
does not match the source used to generate cache"). Fix by removing and
reconfiguring:

```bash
rm -rf solver/build && cmake -S solver -B solver/build -DCMAKE_BUILD_TYPE=Release
```

Connected-pipeline check (before vs. after):

```python
# BEFORE (fragile, broke on the reorg)
ROOT = Path(__file__).resolve().parent.parent
rd = ROOT / "runs" / "out_blast_64_cj"          # wrong: now runs/blast/out_blast_64_cj

# AFTER (robust, category-agnostic)
from paths import run_dir
rd = run_dir("out_blast_64_cj")                 # -> runs/blast/out_blast_64_cj
```

## Related

- `docs/plans/2026-05-23-001-refactor-repo-reorganization-plan.md` — the full reorg plan (how/why)
- [`runs/README.md`](../../../runs/README.md) — run taxonomy + migration tool
- [`postprocessing/README.md`](../../../postprocessing/README.md) — analysis pipeline + resolver
- [`docs/README.md`](../../README.md) — capabilities, numerics, diagnostics, mpi, roadmap, equations PDF
- `docs/solutions/build-errors/mpi-build-fftw3-and-mpi-abi-mismatch-2026-05-17.md` — why the MPI build uses `configure-mpi.sh` + `run_mpi.sh` (ABI scrub)
