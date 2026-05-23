# MPI implementation

The MPI build (`-DBLAST_MPI=ON`) adds a second library `blast_core_mpi` and a
second executable `blast_les_mpi` alongside the serial ones; the serial path is
unaffected. Build into `solver/build_mpi` via `solver/cmake/configure-mpi.sh`,
then run under the ABI-scrub wrapper `solver/run_mpi.sh` (see the build note in
[`solutions/build-errors/`](solutions/build-errors/)).

| Component | Implementation |
|---|---|
| Domain decomposition | `Domain` (`parallel/Domain.cpp`): `MPI_Dims_create` + `MPI_Cart_create` build a 3D Cartesian comm with per-axis periodicity from `BCSet`. Uneven cell counts handled (first `N % Np` ranks get one extra cell). |
| Halo exchange | `Halo` (`parallel/Halo.cpp`): six derived MPI subarray datatypes cached at construction; `exchange()` posts `Isend`/`Irecv` per variable per face, one `Waitall`. 30 messages per call (5 vars × 6 faces). |
| BC dispatch | `apply_bcs(U, bc, Domain&)` (`bc/BC.cpp`): fills only physical-face ghosts, leaving internal partition faces to the halo. |
| Time-step reduction | `max_dt_hyperbolic/viscous(..., MPI_Comm)` (`numerics/RHS.cpp`): `MPI_Allreduce(MIN)` on local dt. |
| Statistics reduction | `velocity_stats` / `dissipation_budget(..., N_global, MPI_Comm)` (`diagnostics/Statistics.cpp`): `MPI_Allreduce(SUM)` on partial sums, divide by global cell count. |
| Snapshot I/O | `HDF5Writer::set_domain` (`io/HDF5Writer.cpp`): single HDF5 file per dump, collective writes (`H5Pset_fapl_mpio`, `COLLECTIVE` hyperslabs from `Domain::global_offset`); scalar writes from rank 0. |
| Spectra (v1) | `velocity_spectrum_mpi` / `helmholtz_decompose_mpi` (`diagnostics/Spectra.cpp`): gather to rank 0 + serial FFTW3. Useful up to ~256³ on a 256 GB node. |
| Spectra (v2, distributed) | `FFT3DPlanMPI` + `*_mpi_dist` (`diagnostics/FFT.cpp`, `Spectra.cpp`): FFTW3-MPI z-slab decomposition; `MPI_Alltoallv` from the 3D Cart layout to FFTW slabs; O(N³/N_ranks) per rank. Cross-checked against v1 to ~1e-17 rel. Scales to the 768³ target. |
| IC global-center | sphere_blast / cj_detonation ICs take an optional `(x_c,y_c,z_c)`; the MPI driver passes the global center so every rank places the IC identically. |

## Verification

- **Bit-exact:** 2- and 4-rank `examples/mpi_smoke.toml` runs are bit-identical to
  serial for KE, tke, M_t, dt at every step. The `test_mpi_bitexact` suite gathers
  a distributed Sod/Sedov run and matches a fresh serial reference to **max |Δ| = 0**.
- **Restart:** collective parallel-HDF5 checkpoint round-trip, max |Δ| = 0 across
  all 5 conserved variables.
- **Distributed spectra:** v1 (gather) vs. v2 (distributed) agree shell-by-shell to
  `rel(E) < 1e-10`.

## Strong scaling (4-core sandbox, 32³ chamber blast)

| Ranks × OMP | Wall | Speedup | Efficiency |
|---|---|---|---|
| 1 × 4 | 8.19 s | 1.00× | — |
| 2 × 2 | 4.73 s | 1.73× | 87 % |
| 4 × 1 | 2.78 s | 2.95× | 74 % |

Above the 50 % strong-scaling bar. On the 9965 these resolutions collapse to
milliseconds; the interesting regime is multi-node MPI for 768³ ensembles. The
scaling drivers live in `scaling/` and write to `runs/scaling/`.
