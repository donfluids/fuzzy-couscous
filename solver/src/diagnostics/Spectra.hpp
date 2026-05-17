#pragma once

#include "core/Grid.hpp"
#include "core/State.hpp"
#include "core/Types.hpp"
#include "diagnostics/FFT.hpp"

#ifdef BLAST_MPI
#include "parallel/Domain.hpp"
#endif

#include <vector>

namespace blast {

// Single shell-averaged spectrum result. k_bins[i] is the bin center
// (in physical wavenumber units 1/length), E[i] is the sum of (1/2)|u_hat|^2
// over the kernel of that shell so that sum_i E[i] dk ~ kinetic energy density
// (when integrated by Parseval). For unit-spacing bins, dk = 2 pi / L.
struct ShellSpectrum {
    std::vector<Real> k;
    std::vector<Real> E;
};

// Energies from a Helmholtz decomposition of the velocity field. K_sol and
// K_dil are spectral integrals of |u_sol|^2/2 and |u_dil|^2/2 respectively.
// Their sum equals the total fluctuation kinetic energy density.
struct HelmholtzResult {
    Real K_sol = 0;
    Real K_dil = 0;
    ShellSpectrum E_sol;
    ShellSpectrum E_dil;
};

// Compute the velocity spectrum E(k) = (1/2) sum_{|k_vec| in shell} |u_hat|^2
// using the conservative state U on the interior region. The discrete spectrum
// is normalized by N^6 so that summing it yields <(1/2) u_i u_i>_volume.
ShellSpectrum velocity_spectrum(const State& U, const Grid& g, FFT3DPlan& plan);

// Helmholtz decomposition in Fourier space:
//   u_hat_dil = (k.u_hat / |k|^2) k
//   u_hat_sol = u_hat - u_hat_dil
// Returns the partitioned energies and per-shell spectra.
HelmholtzResult helmholtz_decompose(const State& U, const Grid& g,
                                    FFT3DPlan& plan);

#ifdef BLAST_MPI
// MPI variants (v1): gather the global velocity field to rank 0 and call
// the serial FFT. Returns populated result on rank 0; empty (k.size()==0)
// on other ranks. NOT scalable to 768^3 -- gathers O(N^3) data and uses
// O(N^3) memory on rank 0. Use velocity_spectrum_mpi_dist below for that.
ShellSpectrum velocity_spectrum_mpi(const State& U, const Grid& global_g,
                                    FFT3DPlan& plan, const Domain& d);
HelmholtzResult helmholtz_decompose_mpi(const State& U, const Grid& global_g,
                                        FFT3DPlan& plan, const Domain& d);

// MPI variants (v2, distributed): use FFTW3-MPI slab decomp. Redistribute
// the 3D Cartesian-decomposed velocity into FFTW's 1D z-slab layout via
// MPI_Alltoallv, run the distributed forward transform, accumulate the
// shell-binned energy locally on each rank's slab, MPI_Allreduce the bins.
// Memory per rank: O(N^3 / Nranks). Same result on every rank.
ShellSpectrum velocity_spectrum_mpi_dist(const State& U, const Grid& global_g,
                                         FFT3DPlanMPI& plan, const Domain& d);
HelmholtzResult helmholtz_decompose_mpi_dist(const State& U, const Grid& global_g,
                                             FFT3DPlanMPI& plan, const Domain& d);
#endif

}  // namespace blast
