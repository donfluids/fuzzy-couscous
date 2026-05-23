# runs/

All simulation outputs live here, one directory per run, grouped by case type.
The run data itself (`out_*/` directories, `*.h5`, `*.csv`, `*.xdmf`, `*.log`) is
**gitignored and regenerable**; only this `README.md`, `MANIFEST.md`, and the
per-category `.gitkeep` placeholders are tracked.

## Categories

| Category | Holds | Rule |
|----------|-------|------|
| `blast/`   | single- and multi-fluid blast runs | name starts `out_blast` |
| `tgv/`     | Taylor–Green vortex runs | name starts `out_tgv` |
| `chamber/` | closed-chamber comparison set (sf / tg / tnt) | name contains `_chamber` |
| `tnt/`     | TNT / JWL free-air and smoke runs | name starts `out_tnt` (non-chamber) |
| `bhr/`     | BHR / RANS turbulence-model runs | name starts `out_bhr` or `out_rans` |
| `scaling/` | strong-scaling sweeps, PGO training, logs | from the `scaling/` drivers |
| `misc/`    | anything matching no rule (e.g. `out_mpi_smoke`) | fallback |

The taxonomy is defined once in `postprocessing/paths.py` (`category_of` /
`run_dir`). Analysis scripts resolve a run with `run_dir("out_blast_128_x")`
rather than hardcoding the category, so re-categorizing touches only that module.

## Conventions

- A solver run writes to `out_dir` from its TOML; configs set
  `out_dir = "runs/<category>/<name>"` so a run lands here directly.
- Run directory names are stable identifiers (e.g. `out_blast_128_budget_seed1`);
  only the parent category changes if a run is reclassified.
- Derived analysis products (`.npz`) live in the top-level `data/`, not here.

## Migration

This tree was consolidated from four former locations (`runs/out_*`,
`solver/out_*`, `solver/runs/out_*`, `scaling/{runs,logs,pgo}_*`) by
`postprocessing/scripts/migrate_runs.py`. `MANIFEST.md` records every old→new
mapping and is the rollback contract:

```bash
python3 postprocessing/scripts/migrate_runs.py --dry-run   # plan only, moves nothing
python3 postprocessing/scripts/migrate_runs.py --apply     # perform the moves
python3 postprocessing/scripts/migrate_runs.py --rollback  # reverse from MANIFEST.md
```
