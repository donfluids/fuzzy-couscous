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

// Cosine/sine-based velocity spectrum for slip-wall domains.
// Slip-wall BCs (ghost mirror with sign-flip on the normal component, see
// bc/BC.cpp:40-93) make the natural spectral basis DCT-II on density / energy
// and on each velocity component's tangential axes, plus DST-II on each
// velocity component's normal axis. Wavenumbers k_n = n*pi/L_axis. The caller
// supplies three pre-built distributed R2R plans, one per velocity component:
//   plan_u : DST x DCT x DCT (kind_x = DST, others = DCT)
//   plan_v : DCT x DST x DCT
//   plan_w : DCT x DCT x DST
// (parameterized in axis order outer-to-inner = z, y, x to match
// R2R3DPlanMPI's constructor signature.)
//
// Returns the shell-binned E(k) summed over the 3 velocity components, with
// bin centers k = b * pi/L_min. Result is identical on every rank.
ShellSpectrum velocity_spectrum_dct_mpi(const State& U, const Grid& global_g,
                                        R2R3DPlanMPI& plan_u,
                                        R2R3DPlanMPI& plan_v,
                                        R2R3DPlanMPI& plan_w,
                                        const Domain& d);

// Helmholtz decomposition in the slip-wall (DCT/DST mixed) basis. Same plan
// triplet as velocity_spectrum_dct_mpi. The decomposition is performed
// entirely in spectral space:
//
//   1. Forward transform u, v, w into their respective DST/DCT mixes.
//   2. Divergence sits in pure DCT_x × DCT_y × DCT_z:
//        div̂[m_x,m_y,m_z] = (m_x pi/L_x) û[m_x-1,m_y,m_z]
//                         + (m_y pi/L_y) v̂[m_x,m_y-1,m_z]
//                         + (m_z pi/L_z) ŵ[m_x,m_y,m_z-1]
//   3. Solve Poisson:  φ̂ = -div̂ / |k_DCT|^2  (skip k=0).
//   4. Recover u_dil = ∇φ component-by-component in their DST/DCT bases:
//        (û_dil_x)[i,j,k] = -((i+1) pi/L_x) φ̂[i+1,j,k]   for i < N-1
//        (similar for v, w on their own DST axes)
//   5. u_sol = u - u_dil per component; bin shells of |u_sol|^2 / 2 and
//      |u_dil|^2 / 2 using each component's own basis wavenumbers.
//
// Inter-rank communication: two single-z-slice exchanges per call (one for
// ŵ along the z-derivative in divergence, one for φ̂ along ∂_z in the
// gradient recovery). Result is identical on every rank.
HelmholtzResult helmholtz_decompose_dct_mpi(const State& U, const Grid& global_g,
                                            R2R3DPlanMPI& plan_u,
                                            R2R3DPlanMPI& plan_v,
                                            R2R3DPlanMPI& plan_w,
                                            const Domain& d);
#endif

}  // namespace blast
