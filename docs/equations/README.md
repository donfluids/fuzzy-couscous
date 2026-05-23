# docs/equations/

`governing-equations.tex` — every governing equation the solver discretizes
(compressible Navier–Stokes, inviscid/viscous fluxes, the three equations of
state including JWL/TNT, multifluid transport, the LES closure, the BHR model,
SSP-RK3, and the diagnostic identities), each section cross-referenced to the
file that implements it.

## Build the PDF

```bash
cd docs/equations
make            # -> governing-equations.pdf
```

The Makefile prefers [`tectonic`](https://tectonic-typesetting.github.io/) — a
single self-contained binary that needs no system TeX install and no root
(it downloads the LaTeX packages it needs on first run). Install it with:

```bash
curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh
# then put the resulting `tectonic` on your PATH
```

If a full TeX distribution is already present, the Makefile falls back to
`latexmk` and then a two-pass `pdflatex` automatically.

The built `governing-equations.pdf` is committed (a `!docs/equations/*.pdf`
exception in `.gitignore`) so readers don't need a TeX toolchain to read it;
rebuild it with `make` after editing the source.
