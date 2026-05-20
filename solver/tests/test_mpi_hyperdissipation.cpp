// MPI pseudospectral hyperdissipation: applies the operator on a single
// Fourier mode and checks that the cellwise RHS matches the analytic
// eigenvalue at every interior cell, across the rank's local sub-block.
// Spectral exactness for periodic single modes means the result should
// match to FFT round-off (~1e-10 at this grid size) regardless of the
// number of ranks the global grid is decomposed into.

#include "bc/BC.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "ic/Canonical.hpp"
#include "numerics/HyperdissipationSpectral.hpp"
#include "parallel/Domain.hpp"
#include "physics/EOS.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdio>

using namespace blast;

namespace {

void fill_density_sin_mode(State& U, const Grid& global, const Domain& d,
                           const IdealGas& eos,
                           Real rho_0, Real A, Real k0, Real p_0) {
    auto off = d.global_offset(global);
    Grid local = d.local_grid(global);
    const int ng = U.ng();
    for (int k = -ng; k < local.nz + ng; ++k)
        for (int j = -ng; j < local.ny + ng; ++j)
            for (int i = -ng; i < local.nx + ng; ++i) {
                const Real x = global.x0 + (off[0] + i + 0.5) * global.dx();
                const Real rho = rho_0 + A * std::sin(k0 * x);
                set_from_primitive(U, i, j, k, eos, rho, 0.0, 0.0, 0.0, p_0);
            }
}

int verify_nabla4_single_mode() {
    const int N = 32;
    Grid global; global.nx = global.ny = global.nz = N;
    global.lx = global.ly = global.lz = 2.0 * M_PI;
    global.x0 = global.y0 = global.z0 = 0.0;
    BCSet bc;
    bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;

    Domain d(MPI_COMM_WORLD, global, bc);
    Grid local = d.local_grid(global);
    IdealGas eos{GammaLaw{}};
    State U(local.nx, local.ny, local.nz);

    const Real rho_0 = 1.0, A = 0.1, k0 = 1.0, p_0 = 1.0;
    fill_density_sin_mode(U, global, d, eos, rho_0, A, k0, p_0);
    apply_bcs(U, bc, d);

    State Rhs(local.nx, local.ny, local.nz);
    for (int v = 0; v < NCONS; ++v) Rhs.fill(v, 0.0);

    HyperdissipationSpectralMpi op(global, d, SpectralBCMode::Periodic);
    const Real nu4 = 0.01;
    op.apply(U, local, nu4, 0.0, Rhs);

    auto off = d.global_offset(global);
    const Real k4 = std::pow(k0, 4);
    Real l2_err_local = 0.0, l2_ref_local = 0.0;
    for (int k = 0; k < local.nz; ++k)
        for (int j = 0; j < local.ny; ++j)
            for (int i = 0; i < local.nx; ++i) {
                const Real x = global.x0 + (off[0] + i + 0.5) * global.dx();
                const Real expected = -nu4 * k4 * A * std::sin(k0 * x);
                const Real got = Rhs[RHO](i, j, k);
                l2_err_local += (got - expected) * (got - expected);
                l2_ref_local += expected * expected;
            }

    Real l2_err = 0.0, l2_ref = 0.0;
    MPI_Allreduce(&l2_err_local, &l2_err, 1, MPI_DOUBLE, MPI_SUM, d.comm());
    MPI_Allreduce(&l2_ref_local, &l2_ref, 1, MPI_DOUBLE, MPI_SUM, d.comm());

    const Real rel = std::sqrt(l2_err / l2_ref);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::printf("[rank0] mpi-spectral nabla4: rel L2 = %.3e (limit 1e-10)\n",
                    rel);
    }
    return (rel < 1e-10) ? 0 : 1;
}

// SlipWall + DCT-DCT-DCT on density: lowest cosine mode along x has
// continuous k = pi/L. Picking L = pi makes k = 1.
int verify_slip_wall_nabla4_density() {
    const int N = 32;
    Grid global; global.nx = global.ny = global.nz = N;
    global.lx = global.ly = global.lz = M_PI;
    global.x0 = global.y0 = global.z0 = 0.0;
    BCSet bc;
    bc.xlo = bc.xhi = BCType::SlipWall;
    bc.ylo = bc.yhi = BCType::SlipWall;
    bc.zlo = bc.zhi = BCType::SlipWall;

    Domain d(MPI_COMM_WORLD, global, bc);
    Grid local = d.local_grid(global);
    IdealGas eos{GammaLaw{}};
    State U(local.nx, local.ny, local.nz);

    const Real rho_0 = 1.0, A = 0.1, p_0 = 1.0;
    auto off = d.global_offset(global);
    const int ng = U.ng();
    for (int k = -ng; k < local.nz + ng; ++k)
        for (int j = -ng; j < local.ny + ng; ++j)
            for (int i = -ng; i < local.nx + ng; ++i) {
                const Real x = global.x0 + (off[0] + i + 0.5) * global.dx();
                const Real rho = rho_0 + A * std::cos(M_PI * x / global.lx);
                set_from_primitive(U, i, j, k, eos, rho, 0.0, 0.0, 0.0, p_0);
            }
    apply_bcs(U, bc, d);

    State Rhs(local.nx, local.ny, local.nz);
    for (int v = 0; v < NCONS; ++v) Rhs.fill(v, 0.0);

    HyperdissipationSpectralMpi op(global, d, SpectralBCMode::SlipWall);
    const Real nu4 = 0.01;
    op.apply(U, local, nu4, 0.0, Rhs);

    const Real k0 = M_PI / global.lx;
    const Real k4 = std::pow(k0, 4);
    Real l2_err_local = 0.0, l2_ref_local = 0.0;
    for (int k = 0; k < local.nz; ++k)
        for (int j = 0; j < local.ny; ++j)
            for (int i = 0; i < local.nx; ++i) {
                const Real x = global.x0 + (off[0] + i + 0.5) * global.dx();
                const Real expected =
                    -nu4 * k4 * A * std::cos(M_PI * x / global.lx);
                const Real got = Rhs[RHO](i, j, k);
                l2_err_local += (got - expected) * (got - expected);
                l2_ref_local += expected * expected;
            }
    Real l2_err = 0, l2_ref = 0;
    MPI_Allreduce(&l2_err_local, &l2_err, 1, MPI_DOUBLE, MPI_SUM, d.comm());
    MPI_Allreduce(&l2_ref_local, &l2_ref, 1, MPI_DOUBLE, MPI_SUM, d.comm());
    const Real rel = std::sqrt(l2_err / l2_ref);
    int rank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) std::printf(
        "[rank0] mpi-spectral slipwall nabla4 (rho/DCT): rel L2 = %.3e "
        "(limit 1e-9)\n", rel);
    return (rel < 1e-9) ? 0 : 1;
}

// SlipWall + DST-DCT-DCT on wall-normal momentum: lowest sine mode along
// x has k = pi/L = 1 for L = pi.
int verify_slip_wall_nabla4_wallnormal_mom() {
    const int N = 32;
    Grid global; global.nx = global.ny = global.nz = N;
    global.lx = global.ly = global.lz = M_PI;
    global.x0 = global.y0 = global.z0 = 0.0;
    BCSet bc;
    bc.xlo = bc.xhi = BCType::SlipWall;
    bc.ylo = bc.yhi = BCType::SlipWall;
    bc.zlo = bc.zhi = BCType::SlipWall;

    Domain d(MPI_COMM_WORLD, global, bc);
    Grid local = d.local_grid(global);
    State U(local.nx, local.ny, local.nz);
    const Real A = 0.1;

    // Set conserved vars directly: rho=1, rhoE=1, only rhou_x is varying.
    for (int v = 0; v < NCONS; ++v) U.fill(v, 0.0);
    auto off = d.global_offset(global);
    const int ng = U.ng();
    for (int k = -ng; k < local.nz + ng; ++k)
        for (int j = -ng; j < local.ny + ng; ++j)
            for (int i = -ng; i < local.nx + ng; ++i) {
                U[RHO ](i, j, k) = 1.0;
                U[RHOE](i, j, k) = 1.0;
                const Real x = global.x0 + (off[0] + i + 0.5) * global.dx();
                U[RHOU](i, j, k) = A * std::sin(M_PI * x / global.lx);
            }
    apply_bcs(U, bc, d);

    State Rhs(local.nx, local.ny, local.nz);
    for (int v = 0; v < NCONS; ++v) Rhs.fill(v, 0.0);

    HyperdissipationSpectralMpi op(global, d, SpectralBCMode::SlipWall);
    const Real nu4 = 0.01;
    op.apply(U, local, nu4, 0.0, Rhs);

    const Real k0 = M_PI / global.lx;
    const Real k4 = std::pow(k0, 4);
    Real l2_err_local = 0.0, l2_ref_local = 0.0;
    for (int k = 0; k < local.nz; ++k)
        for (int j = 0; j < local.ny; ++j)
            for (int i = 0; i < local.nx; ++i) {
                const Real x = global.x0 + (off[0] + i + 0.5) * global.dx();
                const Real expected =
                    -nu4 * k4 * A * std::sin(M_PI * x / global.lx);
                const Real got = Rhs[RHOU](i, j, k);
                l2_err_local += (got - expected) * (got - expected);
                l2_ref_local += expected * expected;
            }
    Real l2_err = 0, l2_ref = 0;
    MPI_Allreduce(&l2_err_local, &l2_err, 1, MPI_DOUBLE, MPI_SUM, d.comm());
    MPI_Allreduce(&l2_ref_local, &l2_ref, 1, MPI_DOUBLE, MPI_SUM, d.comm());
    const Real rel = std::sqrt(l2_err / l2_ref);
    int rank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) std::printf(
        "[rank0] mpi-spectral slipwall nabla4 (rhou_x/DST): rel L2 = %.3e "
        "(limit 1e-9)\n", rel);
    return (rel < 1e-9) ? 0 : 1;
}

int verify_slip_wall_nabla6_density() {
    const int N = 32;
    Grid global; global.nx = global.ny = global.nz = N;
    global.lx = global.ly = global.lz = M_PI;
    global.x0 = global.y0 = global.z0 = 0.0;
    BCSet bc;
    bc.xlo = bc.xhi = BCType::SlipWall;
    bc.ylo = bc.yhi = BCType::SlipWall;
    bc.zlo = bc.zhi = BCType::SlipWall;

    Domain d(MPI_COMM_WORLD, global, bc);
    Grid local = d.local_grid(global);
    IdealGas eos{GammaLaw{}};
    State U(local.nx, local.ny, local.nz);

    const Real rho_0 = 1.0, A = 0.1, p_0 = 1.0;
    auto off = d.global_offset(global);
    const int ng = U.ng();
    for (int k = -ng; k < local.nz + ng; ++k)
        for (int j = -ng; j < local.ny + ng; ++j)
            for (int i = -ng; i < local.nx + ng; ++i) {
                const Real x = global.x0 + (off[0] + i + 0.5) * global.dx();
                const Real rho = rho_0 + A * std::cos(M_PI * x / global.lx);
                set_from_primitive(U, i, j, k, eos, rho, 0.0, 0.0, 0.0, p_0);
            }
    apply_bcs(U, bc, d);

    State Rhs(local.nx, local.ny, local.nz);
    for (int v = 0; v < NCONS; ++v) Rhs.fill(v, 0.0);

    HyperdissipationSpectralMpi op(global, d, SpectralBCMode::SlipWall);
    const Real nu6 = 0.01;
    op.apply(U, local, 0.0, nu6, Rhs);

    const Real k0 = M_PI / global.lx;
    const Real k6 = std::pow(k0, 6);
    Real l2_err_local = 0.0, l2_ref_local = 0.0;
    for (int k = 0; k < local.nz; ++k)
        for (int j = 0; j < local.ny; ++j)
            for (int i = 0; i < local.nx; ++i) {
                const Real x = global.x0 + (off[0] + i + 0.5) * global.dx();
                const Real expected =
                    -nu6 * k6 * A * std::cos(M_PI * x / global.lx);
                const Real got = Rhs[RHO](i, j, k);
                l2_err_local += (got - expected) * (got - expected);
                l2_ref_local += expected * expected;
            }
    Real l2_err = 0, l2_ref = 0;
    MPI_Allreduce(&l2_err_local, &l2_err, 1, MPI_DOUBLE, MPI_SUM, d.comm());
    MPI_Allreduce(&l2_ref_local, &l2_ref, 1, MPI_DOUBLE, MPI_SUM, d.comm());
    const Real rel = std::sqrt(l2_err / l2_ref);
    int rank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) std::printf(
        "[rank0] mpi-spectral slipwall nabla6 (rho/DCT): rel L2 = %.3e "
        "(limit 1e-6)\n", rel);
    return (rel < 1e-6) ? 0 : 1;
}

int verify_nabla6_single_mode() {
    const int N = 32;
    Grid global; global.nx = global.ny = global.nz = N;
    global.lx = global.ly = global.lz = 2.0 * M_PI;
    global.x0 = global.y0 = global.z0 = 0.0;
    BCSet bc;
    bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;

    Domain d(MPI_COMM_WORLD, global, bc);
    Grid local = d.local_grid(global);
    IdealGas eos{GammaLaw{}};
    State U(local.nx, local.ny, local.nz);

    const Real rho_0 = 1.0, A = 0.1, k0 = 1.0, p_0 = 1.0;
    fill_density_sin_mode(U, global, d, eos, rho_0, A, k0, p_0);
    apply_bcs(U, bc, d);

    State Rhs(local.nx, local.ny, local.nz);
    for (int v = 0; v < NCONS; ++v) Rhs.fill(v, 0.0);

    HyperdissipationSpectralMpi op(global, d, SpectralBCMode::Periodic);
    const Real nu6 = 0.01;
    op.apply(U, local, 0.0, nu6, Rhs);

    auto off = d.global_offset(global);
    const Real k6 = std::pow(k0, 6);
    Real l2_err_local = 0.0, l2_ref_local = 0.0;
    for (int k = 0; k < local.nz; ++k)
        for (int j = 0; j < local.ny; ++j)
            for (int i = 0; i < local.nx; ++i) {
                const Real x = global.x0 + (off[0] + i + 0.5) * global.dx();
                const Real expected = -nu6 * k6 * A * std::sin(k0 * x);
                const Real got = Rhs[RHO](i, j, k);
                l2_err_local += (got - expected) * (got - expected);
                l2_ref_local += expected * expected;
            }

    Real l2_err = 0.0, l2_ref = 0.0;
    MPI_Allreduce(&l2_err_local, &l2_err, 1, MPI_DOUBLE, MPI_SUM, d.comm());
    MPI_Allreduce(&l2_ref_local, &l2_ref, 1, MPI_DOUBLE, MPI_SUM, d.comm());

    const Real rel = std::sqrt(l2_err / l2_ref);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::printf("[rank0] mpi-spectral nabla6: rel L2 = %.3e (limit 1e-7)\n",
                    rel);
    }
    // |k|^6 amplification at Nyquist is the same as the serial test: looser
    // tolerance than nabla^4.
    return (rel < 1e-7) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        std::fprintf(stderr, "MPI does not support FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rc = 0;
    rc |= verify_nabla4_single_mode();
    rc |= verify_nabla6_single_mode();
    rc |= verify_slip_wall_nabla4_density();
    rc |= verify_slip_wall_nabla4_wallnormal_mom();
    rc |= verify_slip_wall_nabla6_density();

    int rc_all = 0;
    MPI_Allreduce(&rc, &rc_all, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return rc_all;
}
