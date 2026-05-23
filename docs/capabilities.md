# Solver capabilities

**fuzzy-couscous** is a clean-room compressible LES solver for blast-induced
turbulence in closed chambers, built to defend the claims of *Daniel (2025),
"Blast-induced turbulence in closed chambers."* It targets a single-node AMD EPYC
9965 (192 cores Zen5c, 256 GB) and scales out via MPI to the 768³ production
regime.

For the equations behind each capability, see
[`equations/governing-equations.pdf`](equations/governing-equations.pdf)
(build with `make` in `docs/equations/`).

## Physics regimes

| Regime | What it covers | Key code |
|---|---|---|
| Single-fluid compressible NS | ideal-gas blasts, TGV, forced HIT | `physics/EulerFlux.hpp`, `physics/ViscousFlux.hpp` |
| Two-γ multifluid | variable-density contact (air + products) | `physics/Multifluid.{hpp,cpp}` |
| JWL real explosives | TNT detonation products (non-ideal EOS) | `physics/JWL.hpp`, `physics/MixtureEOS.hpp` |
| CJ detonation IC | Chapman–Jouguet initial state (Williams relations) | `ic/Canonical.cpp` |
| BHR variable-density turbulence | k–ε–a–b model, one- or two-way coupled | `turbulence/BHR.{hpp,cpp}` |
| Closed chamber | slip-wall, energy/mass conserving to round-off | `bc/BC.cpp` |

## Numerical method (summary)

Hybrid 6th-order central / 5th-order WENO5 inviscid reconstruction with a Ducros
shock sensor, 6th-order composed-derivative viscous fluxes, an `ν_h ∇⁴U`
hyperdissipation LES sink, optional Cook/Kawai–Lele localized artificial
diffusivity (LAD), and SSP-RK3 time integration. Full detail in
[`numerics.md`](numerics.md).

## Equations of state

Ideal gas; a marker-selected mixture for blasts — **two-γ** (`G = 1/(γ−1)`) or
**JWL** (TNT, run nondimensionally). One dispatcher
(`physics/MixtureEOS.hpp`) serves the EOS-agnostic flux. See
[`equations/governing-equations.pdf`](equations/governing-equations.pdf) §4.

## Diagnostics (the paper-revision deliverable)

Velocity/Reynolds stats, the solenoidal/dilatational dissipation budget, the
Helmholtz energy split with per-shell spectra, decay-exponent fits with bootstrap
CIs, ensemble averaging, and round-off conservation monitors. Full detail in
[`diagnostics.md`](diagnostics.md).

## Parallelism

OpenMP throughout; optional MPI with 3D Cartesian decomposition, collective
parallel HDF5 I/O, and a distributed FFTW3-MPI spectra path that scales to 768³.
Verified bit-exact serial↔MPI. Full detail in [`mpi.md`](mpi.md).

## Verification & validation

44 serial + 4 MPI test binaries (MMS convergence, exact Riemann, Sedov radius,
TGV decay, Shu–Osher, CJ relations, bit-exact MPI, collective restart, and the
distributed-spectra cross-check). See [`numerics.md`](numerics.md#test-suite) and
the reviewer-concern mapping in the top-level `README.md`.

## Reproducibility

Every run is driven by a TOML config in `solver/examples/` writing to
`runs/<category>/` (see [`../runs/README.md`](../runs/README.md)); analysis is in
`postprocessing/` (see [`../postprocessing/README.md`](../postprocessing/README.md)).
