# CLAUDE.md

Guidance for agents working in **fuzzy-couscous** (a clean-room compressible LES
solver for blast-induced turbulence). Start with `README.md` for build/run/layout.

## Documented solutions

`docs/solutions/` — documented solutions to past problems (bugs, best practices,
conventions, workflow patterns), organized by category with YAML frontmatter
(`module`, `tags`, `problem_type`, `component`). Relevant when implementing or
debugging in documented areas — search it before diving into a new task.

Notable entries: `conventions/repository-conventions-post-reorg-2026-05-23.md`
(where/how to run after the reorg) and `build-errors/` (the MPI FFTW3/ABI build note).

## Key conventions (see the conventions doc above for detail)

- **Run the solver from the repo root** so `out_dir = "runs/<category>/<name>"`
  lands in the unified `runs/` tree.
- Run outputs live in `runs/<category>/` (taxonomy in `postprocessing/paths.py`;
  see `runs/README.md`). Build dirs are `solver/build` (serial) and
  `solver/build_mpi` (MPI, via `solver/cmake/configure-mpi.sh` + `solver/run_mpi.sh`).
- Post-processing is in `postprocessing/` and resolves runs via
  `paths.run_dir(name)` — never hardcode a category (see `postprocessing/README.md`).
- Deep reference docs (capabilities, numerics, diagnostics, mpi, roadmap,
  governing-equations PDF) are under `docs/`.

## Tests

- Solver: `ctest --test-dir solver/build --output-on-failure`
- Post-processing: `python3 -m unittest discover -s postprocessing/tests`
