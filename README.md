# fuzzy-couscous

Compressible LES solver for blast-induced turbulence in closed chambers.

Built clean-room to address the peer-review revisions of *Daniel (2025), Blast
induced turbulence in closed chambers*. Targets a single-node AMD EPYC 9965
(192 cores, 256 GB RAM); C++20 + OpenMP, 6th-order central + Cook (2007)
artificial fluid properties, SSP-RK3.

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
| 1 | AFP shock capture, 1D Sod + Shu-Osher gates | ⏳ |
| 2 | 3D Sedov–Taylor verification | ⏳ |
| 3 | Viscous fluxes + MMS convergence | ⏳ |
| 4 | FFT spectra, Helmholtz split, energy budget | ⏳ |
| 5 | TGV + CBC turbulence validation | ⏳ |
| 6 | Chamber production runs | ⏳ |
| 7 | Paper revision figures | ⏳ |

See `docs/` and the build plan for full details.
