# Documentation

Reference documentation for the **fuzzy-couscous** compressible LES solver.

## Contents

| Doc | What's in it |
|---|---|
| [`capabilities.md`](capabilities.md) | What the solver does — physics regimes, EOS, parallelism, V&V at a glance |
| [`numerics.md`](numerics.md) | Numerical method: schemes, sensors, LES sink, conservation, test suite |
| [`diagnostics.md`](diagnostics.md) | Logged stats, dissipation budget, Helmholtz split, conservation monitors, reviewer-concern mapping |
| [`mpi.md`](mpi.md) | MPI decomposition, halo/I-O, distributed spectra, bit-exact verification, scaling |
| [`equations/`](equations/) | LaTeX source + built PDF of every governing equation |
| [`roadmap.md`](roadmap.md) | Future / deferred work, grouped by strategy track |
| [`solutions/`](solutions/) | Institutional learnings (e.g. the MPI FFTW3/ABI build note) |

## Related, outside `docs/`

- [`../runs/README.md`](../runs/README.md) — the consolidated `runs/<category>/` tree and its taxonomy
- [`../postprocessing/README.md`](../postprocessing/README.md) — the analysis pipeline (`paths.py`, tools vs. scripts)
- [`../STRATEGY.md`](../STRATEGY.md) — product strategy and tracks
- [`plans/`](plans/) — implementation plans
- top-level [`../README.md`](../README.md) — build, requirements, how to run

## Building the equations PDF

```bash
cd docs/equations && make    # -> governing-equations.pdf (needs tectonic or texlive)
```
