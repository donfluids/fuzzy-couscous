# fuzzy-couscous

Compressible LES solver for blast-induced turbulence in closed chambers.

Built clean-room to address the peer-review revisions of *Daniel (2025),
Blast induced turbulence in closed chambers*. Targets a single-node AMD EPYC
9965 (192 cores Zen5c, 256 GB RAM). C++20 + OpenMP, SSP-RK3 time integration,
hybrid 6th-order central / 5th-order WENO with Ducros sensor for shocks,
and a `ν_h ∇⁴ U` hyperdissipation term as the LES sink at high wavenumbers.

## Layout

```
solver/             C++ source, CMake build, GoogleTest suite
solver/examples/    runnable TOML configs (TGV, chamber, paper Case 1, ...)
tools/              Python post-processing (decay fits, spectra plots, ensemble avg)
paper/figures/      committed figures for the manuscript revisions
```

## Build & test

### Serial (OpenMP only)

```bash
cd solver
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Requires: GCC ≥ 13 or Clang ≥ 17, CMake ≥ 3.20, FFTW3 (`libfftw3-dev`), HDF5
(`libhdf5-dev`), spdlog (`libspdlog-dev`), toml++ (`libtomlplusplus-dev`),
GoogleTest, OpenMP.

### MPI (OpenMPI)

```bash
cmake -S solver -B build_mpi -DBLAST_MPI=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_mpi -j
ctest --test-dir build_mpi --output-on-failure   # also runs the MPI halo test at N = 1, 2, 4
mpirun -n 4 ./build_mpi/blast_les_mpi solver/examples/paper_case1.toml
```

Adds: OpenMPI (`libopenmpi-dev`, `openmpi-bin`), parallel HDF5
(`libhdf5-openmpi-dev`), FFTW3-MPI (`libfftw3-mpi-dev`). Builds a second
library `blast_core_mpi` and a second executable `blast_les_mpi` alongside
the serial ones; the serial path is unaffected.

## Run a simulation

### Serial

```bash
./build/blast_les examples/paper_case1.toml          # production: 256³ chamber
./build/blast_les examples/tgv_re1600_64_hyper2.toml # TGV Re=1600 LES on 64³
./build/blast_les examples/chamber_smoke.toml        # 32³ smoke test
```

### MPI

```bash
mpirun -n 4  ./build_mpi/blast_les_mpi examples/paper_case1.toml
mpirun -n 16 ./build_mpi/blast_les_mpi examples/paper_case1.toml
```

3D Cartesian decomposition via `MPI_Dims_create` chooses `(npx, npy, npz)`
automatically; `MPI_Cart_create` builds the topology and sets axis
periodicity from the TOML `[bc]` block. Snapshots are written as a single
HDF5 file per dump (collective parallel HDF5 hyperslabs); the spectra HDF5
is gathered to rank 0 and processed there (production-scale FFTW3-MPI
pencil decomposition is a follow-up).

Outputs go to `<out_dir>` defined in the TOML: HDF5 snapshots + an XDMF
time-series index (ParaView-loadable), a stats CSV, and a spectra HDF5.

## Post-processing

```bash
PY=/path/to/venv/python    # numpy + pandas + h5py + matplotlib
$PY tools/fit_decay.py     out_dir/run_stats.csv  --col tke --plot decay.png
$PY tools/plot_spectra.py  out_dir/run_spectra.h5 -o spectra.png --solenoidal
$PY tools/ensemble_average.py out_a/*_stats.csv out_b/*_stats.csv \
                              --out ens.csv --plot tke,M_t,K_dil
```

## Numerical method

| Component | Implementation |
|---|---|
| Spatial discretization (inviscid) | 6th-order central / 5th-order WENO5 hybrid, flux-difference form, Lax–Friedrichs splitting |
| Shock sensor | Ducros (1999) `θ = (∇·u)²/((∇·u)² + ω² + ε)` + pressure-jump tanh ramp + half-width-2 dilation along the face direction |
| Viscous fluxes | Stokes tensor τ_ij with optional bulk viscosity, energy flux includes τ·u + Fourier heat flux |
| Gradient operator | Composed 6th-order central first derivatives (NGHOST = 6) |
| LES sink | `ν_h ∇⁴ U` hyperdissipation on every conserved variable (`physics.hyper_coeff` knob) |
| Time integration | SSP-RK3 (Gottlieb–Shu) with optional source-term callback for MMS |
| Boundary conditions | Periodic, slip-wall (adiabatic), characteristic outflow |

## Diagnostics (the paper-revision deliverable)

Logged every `stats_every` steps to CSV; the spectra-HDF5 dump optionally
appends shell-averaged E(k), Helmholtz-split E_sol(k) / E_dil(k) per step:

- velocity stats with Reynolds decomposition: `u_rms`, `tke`, `ke_total`,
  `M_t = u_rms / <c>`, `c_mean`, `T_mean`, `rho_mean`, `p_mean`
- dissipation budget: `ε_total = (μ/ρ) ⟨τ_ij ∂u_i/∂x_j⟩`,
  `ε_sol = ν ⟨|ω|²⟩`, `ε_dil = (4/3) ν ⟨(∇·u)²⟩`
- enstrophy `⟨|ω|²⟩` and dilatation-squared `⟨(∇·u)²⟩` independently
- Helmholtz solenoidal/dilatational kinetic-energy split + per-shell spectra

## Test suite (44 serial + 3 MPI rank counts, all passing)

| Suite | What it verifies | Wall (9965 est.) |
|---|---|---|
| Field3D (7), Config (2), Stencil (2), BC (3) | allocation, NUMA touch, TOML parsing, 6th-order stencil, periodic / outflow / slip-wall ghost fill | < 1 s |
| AdvectSmooth, Sod1D, Sedov3D, ViscousMMS | RK3 third-order convergence on smooth periodic flow; exact Riemann; analytic shock radius; composed-derivative viscous Laplacian 6th-order | ~2 s |
| Spectrum, Helmholtz, VelocityStats, Dissipation | Parseval; single-mode binning; pure-solenoidal / pure-irrotational projection; M_t / ε_sol / ε_dil identities | < 1 s |
| TGV (initial enstrophy + decay), ChamberSmoke, CJDetonation, Restart | viscous decay, slip-wall closed-chamber blast smoke, exact CJ relations, checkpoint bit-exactness | ~5 s |
| **MMS3D** (3) — entropy wave, Yee–Sandham–Djomehri vortex, viscous-NS with analytic source | end-to-end Navier–Stokes accuracy on 32³ → 48³ → 64³ | ~30 s |
| **Hyperdissipation** (3) — operator value, eigenvalue decay, off-when-disabled | discrete `∇⁴` matches analytic `k⁴` damping; SSP-RK3 integration preserves the exp(−λ t) eigenvalue decay; zero coeff bit-exact identity | ~10 s |
| **SlipWall** (2) — mass+energy conservation, acoustic round-trip | inviscid slip walls conserve mass and total energy to < 1e-10 relative; small-amplitude acoustic pulse returns to its starting position with > 50% amplitude after one round-trip, no negative-density ringing | ~5 s |
| **ShuOsher1D** (2) — post-shock oscillations, refinement | Mach-3 shock interacting with sinusoidal density retains the post-shock fine-scale oscillations; 200-cell vs 800-cell self-converged reference L1 < 0.25 | ~3 s |
| **MPI halo** (1 binary × 3 rank counts) — periodic exchange + face count + cell sum | analytic continuation across 2/4/8-rank partitions to round-off; physical-face count matches `2(npx npy + npx npz + npy npz)`; sum of local cell counts equals global Nx·Ny·Nz with non-divisible factors | < 1 s |
| **MPI bit-exact** (1 binary × 3 rank counts) — Sod 1D & Sedov 3D | Distributed run gathered to rank 0 matches a fresh serial reference of the same problem to **bit precision** (max |Δ| = 0) at K = 20 / 15 steps. Stresses halo exchange, BC-on-physical-only faces, dt Allreduce, and RHS stencil across partitions. | < 5 s |

Total runtime on this sandbox (4 cores, single NUMA): ~95 minutes,
dominated by MMS3D + SlipWall. On the 9965 with 192 threads the suite
should run in well under a minute.

## MPI implementation details

| Component | Implementation |
|---|---|
| Domain decomposition | `Domain` (parallel/Domain.cpp): `MPI_Dims_create` + `MPI_Cart_create` build a 3D Cartesian comm with per-axis periodicity from `BCSet`. Uneven cell counts handled (first `N % Np` ranks get one extra cell). |
| Halo exchange | `Halo` (parallel/Halo.cpp): six derived MPI subarray datatypes cached at construction; `exchange()` posts `MPI_Isend` + `MPI_Irecv` per variable per face and finishes with one `MPI_Waitall`. 30 messages per call (5 conserved variables × 6 faces). |
| BC dispatch | `apply_bcs(U, bc, Domain&)` (bc/BC.cpp): only fills physical-face ghost cells, leaving internal partition faces to the halo exchange. |
| Time step reduction | `max_dt_hyperbolic(..., MPI_Comm)` and `max_dt_viscous(..., MPI_Comm)` (numerics/RHS.cpp): `MPI_Allreduce(MIN)` on local dt. |
| Statistics reduction | `velocity_stats(..., N_global, MPI_Comm)` and `dissipation_budget(..., N_global, MPI_Comm)` (diagnostics/Statistics.cpp): `MPI_Allreduce(SUM)` on partial sums, then divide by global cell count. |
| Snapshot I/O | `HDF5Writer::set_domain(Domain*)` (io/HDF5Writer.cpp): single HDF5 file per dump, collective writes via `H5Pset_fapl_mpio` and `H5Pset_dxpl_mpio(...COLLECTIVE)` with hyperslabs from `Domain::global_offset`. Independent-mode scalar writes from rank 0 only. |
| Spectra (v1) | `velocity_spectrum_mpi` / `helmholtz_decompose_mpi` (diagnostics/Spectra.cpp): gather to rank 0 + serial FFTW3 on the global grid. **Not scalable to 768³**; FFTW3-MPI pencil decomp is a flagged TODO. |
| IC global-center | sphere_blast / cj_detonation ICs take optional explicit `(x_c, y_c, z_c)`; the MPI driver passes the global domain center so every rank places the IC at the same physical location. |

End-to-end verification: 2-rank and 4-rank `examples/mpi_smoke.toml` runs
are **bit-identical to serial** for KE, tke, M_t, dt at every step.

## TGV at Re=1600 on 64³ — Kolmogorov recovery

`examples/tgv_re1600_64_hyper2.toml` runs the classical Taylor–Green vortex
to t = 12 with `ν = 1/1600` and `ν_h = 2.5e-5`. The shell-averaged spectrum
develops a clean `k^(-5/3)` inertial range in `k ∈ [4, 10]` for `t ≥ 8`;
local slope settles to **−1.5 to −1.7** in that window. tke decays from
0.125 → 0.043 (~65 % drop), peak dissipation at t ≈ 9, ε_dil / ε_sol stays
below 10⁻⁵ throughout (purely solenoidal as required for the TGV symmetry
class). Figure under `paper/figures/tgv_re1600_64_kolmogorov.{png,pdf}`.

The hyperdissipation coefficient that worked at 64³ (`2.5e-5`) is `Δx²`
scaled from the 32³ value (`1e-4`) — not `Δx⁴` as a naive eigenvalue
argument would suggest. Empirically, fixed `ν_eff = ν_h k_max²` was the
stability invariant; matching dissipation rate at Nyquist instead lets the
cascade outpace the sink.

## Mapping to the paper revisions

| Reviewer concern | Code path |
|---|---|
| M1 effective Reynolds | `dissipation_budget` separates resolved viscous ε; numerical sink quantified via `hyper_coeff` |
| M2 grid convergence | re-run any example at varying `nx` |
| M3 decay exponent + CI | `tools/fit_decay.py` (2000-sample bootstrap) |
| M4 solenoidal/dilatational | `helmholtz_decompose` + per-step `K_sol`, `K_dil`, `ε_sol`, `ε_dil` |
| M5 / M8 tke vs EKE | `velocity_stats` reports both with mean removal |
| M6 ensembles | `ensemble_seed` + multi-mode random IC + `tools/ensemble_average.py` |
| M7 early-time sanity | configurable `snapshot_every`; pre/post first-reflection capture |
| M9 CJ initial condition | `ic_cj_detonation_3d` (Williams exact relations, validated to 1e-6) |
