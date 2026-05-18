#pragma once

#include "core/Grid.hpp"
#include "core/State.hpp"
#include "core/Types.hpp"

#ifdef BLAST_MPI
#include <mpi.h>
#endif

#include <array>
#include <complex>
#include <random>
#include <vector>

namespace blast {

// Eswaran-Pope (1988) / Alvelius (1999) stochastic spectral forcing for
// statistically stationary homogeneous isotropic turbulence.
//
// State: one complex 3-vector per low-k Fourier mode in the integer shell
// [k_lo, k_hi]. Each mode evolves under an Ornstein-Uhlenbeck process with
// correlation time T_corr. Amplitudes are projected onto the plane
// perpendicular to k (helical basis e1, e2) so the resulting force is
// exactly solenoidal in the continuum. Force is evaluated in physical
// space by direct summation (a few hundred modes at k_hi <= 4; cheaper
// than an extra distributed FFT in the RHS loop).
//
// Every step the amplitude is uniformly rescaled so the global injection
// power <rho u . f> / <rho> equals the user-specified eps_target. That
// makes the dissipation rate at equilibrium predictable a priori.
//
// MPI: every rank carries the SAME mode list with the SAME OU state (same
// RNG seed + identical update). Each rank then evaluates f at its local
// physical coordinates. Forcing is therefore bit-exact under any
// 3D-Cartesian decomposition. Only the global power measurement uses
// MPI_Allreduce(SUM); apply itself does no communication.
class SpectralForcing {
public:
    struct Params {
        // Integer wavenumber shell (in fundamental units of 2 pi / L)
        // where forcing modes live. Default 1..3 = energy-containing range.
        int  k_lo = 1;
        int  k_hi = 3;
        // Target mean injection power per unit mass; equals the dissipation
        // rate at statistical stationarity.
        Real eps_target = 0.1;
        // Ornstein-Uhlenbeck correlation time. Set this comparable to the
        // expected eddy-turnover time L / u_rms.
        Real T_corr = 1.0;
        // RNG seed; identical across ranks so the OU evolution is shared.
        int  seed = 12345;
    };

    SpectralForcing(const Grid& global_grid, const Params& p);

    // Advance the OU random state by one substep dt. Same evolution on
    // every rank (same RNG seed + identical step).
    void evolve_ou(Real dt);

    // Compute the raw force field at every local cell, measure <rho u . f>
    // globally, rescale so that mean injection power equals eps_target,
    // then add dt * rho * f to momentum and dt * rho * (u . f) to total
    // energy. `local` is the rank's local grid (with the per-rank origin),
    // not the global one. comm = MPI_COMM_NULL for serial calls.
    void apply(State& U, const Grid& local, Real dt
#ifdef BLAST_MPI
               , MPI_Comm comm = MPI_COMM_NULL
#endif
               );

    // Number of forced Fourier modes (Hermitian half of integer lattice
    // intersecting the shell).
    int num_modes() const { return static_cast<int>(modes_.size()); }

    // Last measured global injection power after rescaling. Should equal
    // eps_target to round-off when the velocity field is non-degenerate.
    Real last_inject_power() const { return last_eps_; }

private:
    struct Mode {
        Real kx, ky, kz;
        int  mx, my, mz;          // integer wavenumber indices (for trig table lookup)
        Real kmag;
        Real e1[3], e2[3];        // unit basis vectors perpendicular to k
        std::complex<Real> a1;    // OU amplitude along e1
        std::complex<Real> a2;    // OU amplitude along e2
    };

    Params p_;
    std::vector<Mode> modes_;
    std::mt19937 rng_;
    Real last_eps_ = 0.0;

    // Per-axis trig tables. For each integer wavenumber m in [-k_hi, +k_hi]
    // (offset by k_hi so m=0 is at index k_hi), store cos(m * 2pi/L * x_i)
    // and sin(...) for every local cell index i. Built once on the first
    // apply() call; reused on every subsequent call.
    //
    // Layout: cos_table_x_[(m + k_hi) * nx_local + i].
    //
    // Replaces ~3 billion cos/sin evaluations per step at 256^3 / 89 modes
    // with table lookups + 2 trig-sum identities per cell.
    mutable std::vector<Real> cos_table_x_, sin_table_x_;
    mutable std::vector<Real> cos_table_y_, sin_table_y_;
    mutable std::vector<Real> cos_table_z_, sin_table_z_;
    mutable int tab_nx_ = 0, tab_ny_ = 0, tab_nz_ = 0;
    mutable Real tab_x0_ = 0, tab_y0_ = 0, tab_z0_ = 0;
    mutable Real tab_dx_ = 0, tab_dy_ = 0, tab_dz_ = 0;

    void ensure_trig_tables(const Grid& local) const;
};

}  // namespace blast
