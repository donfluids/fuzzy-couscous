# Post-processing tools

Python scripts that consume the outputs of `blast_les`.

## Setup

Use a venv with up-to-date numpy/pandas/h5py/matplotlib:

```bash
python3 -m venv /tmp/blast-venv
/tmp/blast-venv/bin/pip install numpy pandas h5py matplotlib
PYTHON=/tmp/blast-venv/bin/python3
```

## Scripts

### `fit_decay.py`

Bootstrap-CI power-law fit of any `<column>` in `<run>_stats.csv` against time.
Default column is `tke`.

```bash
$PYTHON tools/fit_decay.py solver/build/out_chamber_smoke/chamber_smoke_stats.csv \
    --col tke --t-min 0.05 --plot tke_fit.png
```

Output: `tke = A * t^(-n)` with the 95% CI on `n` from a 2000-sample residual
bootstrap. Use `--col K_sol`, `--col K_dil`, `--col M_t`, etc. to study other
quantities.

### `plot_spectra.py`

Overlays shell-averaged spectra E(k) at several times from
`<run>_spectra.h5`. Optional `--solenoidal` adds dashed/dotted curves for the
Helmholtz components (solid total, `--` sol, `:` dil), annotates the
`K_sol/K_tot` and `K_dil/K_tot` *energy* fractions per time, and adds a second
panel with the per-component local-slope (*turbulence*). The slope panel is
smoothed by a log-k local fit (`--slope-smooth`, ln-k half-width) and masked
below the dissipation tail (`--slope-floor`, fraction of the component peak) so
the linear-k bin noise does not swamp it; this affects the *plot* only, not the
slope numbers reported by `spectrum_shape.py`.

```bash
$PYTHON tools/plot_spectra.py solver/build/out_chamber_smoke/chamber_smoke_spectra.h5 \
    -o spectra.png --solenoidal --reference-slope=-5/3
```

Steps to plot can be selected explicitly with `--steps 50,150,300,600`.

### `spectrum_shape.py`

Reports the two distinct axes of a spectrum from `<run>_spectra.h5`, separately:
the **energy** (integrated `K_sol`/`K_dil` magnitudes and fractions) and the
**turbulence** (local slope + contiguous power-law range, per component), so
dilatational *energy* (how much) is never read as dilatational *turbulence*
(whether it cascades). Shared shape/energy helpers used by `plot_spectra.py`
and `scripts/spectrum_components.py`.

```bash
$PYTHON tools/spectrum_shape.py runs/out_blast_128_budget_seed1/blast_128_budget_seed1_spectra.h5 \
    --t0 0.01 --t1 0.02 --target-slope -5/3
```
