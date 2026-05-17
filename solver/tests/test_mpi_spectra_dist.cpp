// Distributed FFTW3-MPI spectrum / Helmholtz cross-check against the
// gather-to-rank-0 reference. Both code paths exist in the MPI build;
// running them on the same analytic multi-mode velocity field on a
// 3D-Cartesian decomposition and comparing shell-by-shell agreement to
// FFT round-off validates the Alltoallv redistribution + slab transform.

#include "bc/BC.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "diagnostics/FFT.hpp"
#include "diagnostics/Spectra.hpp"
#include "parallel/Domain.hpp"
#include "physics/EOS.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace blast;

namespace {

// Multi-mode periodic velocity defined globally. Density and pressure are
// uniform so u_i = (rho u_i) / rho exactly. We pick wavenumbers and phases
// that span a few shells (k = 1..6) so the spectrum is non-trivial. The
// field is generally NOT divergence-free, which is intentional: it exercises
// both the solenoidal and dilatational paths of helmholtz_decompose.
void fill_analytic_field(State& U, const Grid& global,
                         const std::array<long long, 3>& off) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz();
    auto& rho  = U[RHO];
    auto& rhou = U[RHOU];
    auto& rhov = U[RHOV];
    auto& rhow = U[RHOW];
    auto& rhoE = U[RHOE];

    const Real rho0 = 1.0;
    const Real p0   = 1.0;
    const Real gam  = 1.4;

    const Real Lx = global.lx, Ly = global.ly, Lz = global.lz;
    const Real two_pi_Lx = 2.0 * M_PI / Lx;
    const Real two_pi_Ly = 2.0 * M_PI / Ly;
    const Real two_pi_Lz = 2.0 * M_PI / Lz;

    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const long long i_g = off[0] + i;
                const long long j_g = off[1] + j;
                const long long k_g = off[2] + k;
                const Real x = global.x0 + (i_g + 0.5) * global.dx();
                const Real y = global.y0 + (j_g + 0.5) * global.dy();
                const Real z = global.z0 + (k_g + 0.5) * global.dz();

                const Real u =  std::sin(2 * two_pi_Lx * x) * std::cos(1 * two_pi_Ly * y)
                              + 0.3 * std::cos(4 * two_pi_Lx * x + 0.7);
                const Real v =  std::cos(2 * two_pi_Lx * x) * std::sin(3 * two_pi_Lz * z)
                              - 0.2 * std::sin(1 * two_pi_Ly * y + 1.3);
                const Real w =  std::sin(1 * two_pi_Ly * y) * std::sin(2 * two_pi_Lz * z)
                              + 0.5 * std::cos(5 * two_pi_Lx * x + 0.1);

                rho(i, j, k)  = rho0;
                rhou(i, j, k) = rho0 * u;
                rhov(i, j, k) = rho0 * v;
                rhow(i, j, k) = rho0 * w;
                const Real ke = 0.5 * rho0 * (u*u + v*v + w*w);
                rhoE(i, j, k) = p0 / (gam - 1.0) + ke;
            }
}

State make_state(int N, const Grid& global, const Domain& d, Grid& local_out) {
    local_out = d.local_grid(global);
    State U(local_out.nx, local_out.ny, local_out.nz);
    auto off = d.global_offset(global);
    fill_analytic_field(U, global, off);
    BCSet bc; bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic; bc.zlo = bc.zhi = BCType::Periodic;
    apply_bcs(U, bc);
    return U;
}

}  // namespace

// Compare velocity_spectrum_mpi (gather-to-rank-0) vs.
// velocity_spectrum_mpi_dist (FFTW3-MPI). Both must agree to FFT precision.
static int test_spectrum_matches() {
    const int N = 32;
    Grid global; global.nx = global.ny = global.nz = N;
    global.lx = global.ly = global.lz = 2.0 * M_PI;
    global.x0 = global.y0 = global.z0 = 0.0;
    BCSet bc; bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic; bc.zlo = bc.zhi = BCType::Periodic;
    Domain d(MPI_COMM_WORLD, global, bc);
    Grid local;
    State U = make_state(N, global, d, local);

    FFT3DPlan      plan_serial(N, N, N);
    FFT3DPlanMPI   plan_dist(N, N, N, d.comm());

    ShellSpectrum sp_ref  = velocity_spectrum_mpi(U, global, plan_serial, d);
    ShellSpectrum sp_dist = velocity_spectrum_mpi_dist(U, global, plan_dist, d);

    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int rc = 0;
    if (rank == 0) {
        if (sp_ref.E.size() != sp_dist.E.size()) {
            std::fprintf(stderr, "size mismatch: ref=%zu dist=%zu\n",
                         sp_ref.E.size(), sp_dist.E.size());
            return 1;
        }
        Real max_abs = 0.0, sum_ref = 0.0;
        for (std::size_t b = 0; b < sp_ref.E.size(); ++b) {
            max_abs = std::max(max_abs, std::fabs(sp_ref.E[b] - sp_dist.E[b]));
            sum_ref += sp_ref.E[b];
        }
        const Real rel = max_abs / std::max(sum_ref, 1e-30);
        std::printf("[rank0] spectrum: K_ref=%.6e max|dE|=%.3e rel=%.3e\n",
                    sum_ref, max_abs, rel);
        if (rel > 1e-10) {
            std::fprintf(stderr, "spectrum mismatch rel=%g > 1e-10\n", rel);
            rc = 1;
        }
    }
    int rc_all = 0;
    MPI_Allreduce(&rc, &rc_all, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return rc_all;
}

// Compare helmholtz_decompose_mpi vs. helmholtz_decompose_mpi_dist on the
// same analytic field. Solenoidal/dilatational energies + spectra must match.
static int test_helmholtz_matches() {
    const int N = 32;
    Grid global; global.nx = global.ny = global.nz = N;
    global.lx = global.ly = global.lz = 2.0 * M_PI;
    global.x0 = global.y0 = global.z0 = 0.0;
    BCSet bc; bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic; bc.zlo = bc.zhi = BCType::Periodic;
    Domain d(MPI_COMM_WORLD, global, bc);
    Grid local;
    State U = make_state(N, global, d, local);

    FFT3DPlan      plan_serial(N, N, N);
    FFT3DPlanMPI   plan_dist(N, N, N, d.comm());

    HelmholtzResult ref  = helmholtz_decompose_mpi(U, global, plan_serial, d);
    HelmholtzResult dist = helmholtz_decompose_mpi_dist(U, global, plan_dist, d);

    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int rc = 0;
    if (rank == 0) {
        const Real Kref  = ref.K_sol + ref.K_dil;
        const Real Kdist = dist.K_sol + dist.K_dil;
        const Real rel_K = std::fabs(Kref - Kdist) / std::max(Kref, 1e-30);
        std::printf("[rank0] helmholtz: K_sol ref=%.6e dist=%.6e  "
                    "K_dil ref=%.6e dist=%.6e  rel(K)=%.3e\n",
                    ref.K_sol, dist.K_sol, ref.K_dil, dist.K_dil, rel_K);
        if (rel_K > 1e-10) rc = 1;

        Real max_sol = 0, max_dil = 0;
        for (std::size_t b = 0; b < ref.E_sol.E.size(); ++b) {
            max_sol = std::max(max_sol, std::fabs(ref.E_sol.E[b] - dist.E_sol.E[b]));
            max_dil = std::max(max_dil, std::fabs(ref.E_dil.E[b] - dist.E_dil.E[b]));
        }
        std::printf("[rank0] max|dE_sol|=%.3e max|dE_dil|=%.3e\n",
                    max_sol, max_dil);
        const Real norm = std::max(Kref, 1e-30);
        if (max_sol > 1e-10 * norm) rc = 1;
        if (max_dil > 1e-10 * norm) rc = 1;
    }
    int rc_all = 0;
    MPI_Allreduce(&rc, &rc_all, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return rc_all;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rc = 0;
    rc |= test_spectrum_matches();
    rc |= test_helmholtz_matches();

    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0 && rc == 0) std::printf("[OK] spectra_dist match reference\n");

    MPI_Finalize();
    return rc;
}
