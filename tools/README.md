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
Helmholtz components.

```bash
$PYTHON tools/plot_spectra.py solver/build/out_chamber_smoke/chamber_smoke_spectra.h5 \
    -o spectra.png --solenoidal --reference-slope -5/3
```

Steps to plot can be selected explicitly with `--steps 50,150,300,600`.
