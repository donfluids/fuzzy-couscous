# postprocessing/

All Python post-processing for fuzzy-couscous: diagnostics, spectra, decay fits,
budgets, movies, and scaling analysis. Separated from the solver and the run
outputs so analysis code lives in one place.

## Layout

```
postprocessing/
  paths.py            shared resolver: REPO_ROOT, RUNS, DATA, FIGS, run_dir(), category_of()
  tools/              reusable utilities (path taken as a CLI argument)
                        fit_decay.py, plot_spectra.py, ensemble_average.py,
                        spectrum_shape.py, plot_tgv_spectra.py, plot_forced_hit.py
  scripts/            case-specific analyses (resolve runs by name via run_dir)
    scaling/          strong-scaling / PGO sweep analysis + parse_steps.py
    migrate_runs.py   one-shot runs-consolidation tool (dry-run/apply/rollback)
  tests/              unittest suite (resolver, migration, import smoke test)
```

`tools/` vs `scripts/`: **tools** are general-purpose and take an explicit
`out_dir`/file path on the command line (reusable across runs); **scripts** are
tied to specific cases and resolve their inputs by run name.

## The path resolver

Every script that touches run data imports from `paths` instead of hardcoding
locations:

```python
from paths import REPO_ROOT, RUNS, DATA, FIGS, run_dir
rd = run_dir("out_blast_128_budget_seed1")   # -> <repo>/runs/blast/out_blast_128_budget_seed1
```

`run_dir(name)` hides the run categorization (`runs/<category>/<name>`), so a run
can be reclassified by editing only `paths.py`. `REPO_ROOT` is found by walking
up to a `.git` / `solver/CMakeLists.txt` marker, so scripts work from any depth.
See `../runs/README.md` for the category taxonomy.

## Running a script

Scripts add `postprocessing/`, `tools/`, and `scripts/` to `sys.path` themselves
(via a small bootstrap), so they run directly:

```bash
PY=python3   # needs numpy, scipy, h5py, pandas, matplotlib (see ../requirements.txt)
$PY postprocessing/tools/fit_decay.py runs/blast/out_blast_128_budget_seed1/blast_128_budget_seed1_stats.csv --col tke
$PY postprocessing/scripts/compare_mf_sg_spectra.py 64
$PY postprocessing/scripts/migrate_runs.py --dry-run
```

Figures are written to the top-level `figs/`.

## Tests

```bash
python3 -m unittest discover -s postprocessing/tests
```

Covers the resolver, the runs migration (on a fixture tree), and a smoke test
that every script compiles, carries no stale path anchor, and imports cleanly.
