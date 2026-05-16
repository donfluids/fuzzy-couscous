# fuzzy-couscous

Compressible LES solver for blast-induced turbulence in closed chambers.

Built clean-room to address the peer-review revisions of *Daniel (2025), Blast
induced turbulence in closed chambers*. Targets a single-node AMD EPYC 9965
(192 cores, 256 GB RAM); C++20 + OpenMP, SSP-RK3, hybrid 6th-order central /
5th-order WENO with Ducros sensor (no per-problem coefficient tuning).

## Layout

```
solver/        C++ source, CMake build, GoogleTest suite
tools/         Python post-processing (planned)
validation/    Canonical regression tests (planned)
docs/          Numerics / diagnostics notes (planned)
paper/         Manuscript revisions (planned)
```

## Build & test

```bash
cd solver
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Requires: GCC ≥ 13 or Clang ≥ 17, CMake ≥ 3.20, FFTW3, HDF5, spdlog,
toml++, GoogleTest, OpenMP.

## Current status

| Phase | Component | Status |
|------:|-----------|--------|
| 0 | CMake / Field3D / Config / Log | ✅ |
| 1 | 6th-order stencils, Euler flux, SSP-RK3, BCs (periodic/outflow/slip) | ✅ |
| 1 | Hybrid central / WENO5 + Ducros, 1D Sod validation (positivity + L1 + refinement) | ✅ |
| 1 | Shu-Osher gate | ⏳ |
| 2 | 3D Sedov–Taylor verification | ⏳ |
| 3 | Viscous fluxes + MMS convergence | ⏳ |
| 4 | FFT spectra, Helmholtz split, energy budget | ⏳ |
| 5 | TGV + CBC turbulence validation | ⏳ |
| 6 | Chamber production runs | ⏳ |
| 7 | Paper revision figures | ⏳ |

See `docs/` and the build plan for full details.
