// MPI bit-exact verification for SpectralForcing. Same RNG seed on every
// rank means every rank carries the SAME OU state; each rank then evaluates
// the force at its local cells (using global coordinates). Result: the
// distributed update must match a serial reference of the same problem to
// floating-point round-off, exactly like test_mpi_bitexact for the RHS.

#include "bc/BC.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "ic/Canonical.hpp"
#include "parallel/Domain.hpp"
#include "physics/EOS.hpp"
#include "physics/Forcing.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace blast;

namespace {

State make_local(int N, const Grid& global, const Domain& d,
                 Grid& local_out) {
    local_out = d.local_grid(global);
    State U(local_out.nx, local_out.ny, local_out.nz);
    auto off = d.global_offset(global);
    IdealGas eos{GammaLaw{}};

    // Seed the state with a smooth analytic field consistent across ranks.
    auto& rho = U[RHO];
    auto& mx = U[RHOU]; auto& my = U[RHOV]; auto& mz = U[RHOW];
    auto& en = U[RHOE];
    const Real gam = 1.4;
    for (int k = 0; k < local_out.nz; ++k)
        for (int j = 0; j < local_out.ny; ++j)
            for (int i = 0; i < local_out.nx; ++i) {
                const Real x = global.x0 + (off[0] + i + 0.5) * global.dx();
                const Real y = global.y0 + (off[1] + j + 0.5) * global.dy();
                const Real z = global.z0 + (off[2] + k + 0.5) * global.dz();
                const Real u = 0.1 * std::sin(x) * std::cos(y);
                const Real v = 0.1 * std::cos(x) * std::sin(z);
                const Real w = 0.1 * std::sin(y) * std::sin(z);
                rho(i, j, k) = 1.0;
                mx(i, j, k)  = u;
                my(i, j, k)  = v;
                mz(i, j, k)  = w;
                const Real ke = 0.5 * (u*u + v*v + w*w);
                en(i, j, k) = 1.0 / (gam - 1.0) + ke;
            }
    BCSet bc;
    bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;
    apply_bcs(U, bc, d);
    return U;
}

// Gather rank-local interior cells of one conserved variable into a
// global buffer on rank 0.
void gather_field(const State& U, int v, const Grid& global, const Domain& d,
                  std::vector<Real>& out_global) {
    const int nx_l = U.nx(), ny_l = U.ny(), nz_l = U.nz();
    const auto off = d.global_offset(global);
    const int nxg = global.nx, nyg = global.ny;

    const std::size_t Nlocal =
        static_cast<std::size_t>(nx_l) * ny_l * nz_l;
    std::vector<Real> local(Nlocal);
    auto& f = U[v];
    for (int k = 0; k < nz_l; ++k)
        for (int j = 0; j < ny_l; ++j)
            for (int i = 0; i < nx_l; ++i) {
                const std::size_t li = static_cast<std::size_t>(i)
                    + nx_l * (static_cast<std::size_t>(j) + ny_l * k);
                local[li] = f(i, j, k);
            }

    if (d.rank() == 0) {
        out_global.assign(
            static_cast<std::size_t>(global.nx) * global.ny * global.nz, 0.0);
        // Self-fill rank 0.
        for (int k = 0; k < nz_l; ++k)
            for (int j = 0; j < ny_l; ++j)
                for (int i = 0; i < nx_l; ++i) {
                    const std::size_t li = static_cast<std::size_t>(i)
                        + nx_l * (static_cast<std::size_t>(j) + ny_l * k);
                    const std::size_t gi =
                        static_cast<std::size_t>(off[0] + i)
                        + nxg * (static_cast<std::size_t>(off[1] + j)
                                 + nyg * (off[2] + k));
                    out_global[gi] = local[li];
                }
        for (int r = 1; r < d.size(); ++r) {
            int meta[6];
            MPI_Recv(meta, 6, MPI_INT, r, 2000, d.comm(), MPI_STATUS_IGNORE);
            const int ox = meta[0], oy = meta[1], oz = meta[2];
            const int ex = meta[3], ey = meta[4], ez = meta[5];
            const std::size_t rsz = static_cast<std::size_t>(ex) * ey * ez;
            std::vector<Real> rbuf(rsz);
            MPI_Recv(rbuf.data(), static_cast<int>(rsz), MPI_DOUBLE, r, 2001,
                     d.comm(), MPI_STATUS_IGNORE);
            for (int k = 0; k < ez; ++k)
                for (int j = 0; j < ey; ++j)
                    for (int i = 0; i < ex; ++i) {
                        const std::size_t li = static_cast<std::size_t>(i)
                            + ex * (static_cast<std::size_t>(j) + ey * k);
                        const std::size_t gi = static_cast<std::size_t>(ox + i)
                            + nxg * (static_cast<std::size_t>(oy + j)
                                     + nyg * (oz + k));
                        out_global[gi] = rbuf[li];
                    }
        }
    } else {
        int meta[6] = {static_cast<int>(off[0]), static_cast<int>(off[1]),
                       static_cast<int>(off[2]), nx_l, ny_l, nz_l};
        MPI_Send(meta, 6, MPI_INT, 0, 2000, d.comm());
        MPI_Send(local.data(), static_cast<int>(Nlocal), MPI_DOUBLE, 0, 2001,
                 d.comm());
    }
}

}  // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rc_all = 0;
    {
    const int N = 24;
    Grid global;
    global.nx = global.ny = global.nz = N;
    global.lx = global.ly = global.lz = 2.0 * M_PI;
    global.x0 = global.y0 = global.z0 = 0.0;

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;

    Domain d(MPI_COMM_WORLD, global, bc);
    Grid local;
    State U = make_local(N, global, d, local);

    SpectralForcing::Params p;
    p.k_lo = 1; p.k_hi = 3; p.eps_target = 0.05;
    p.T_corr = 1.0; p.seed = 271828;
    SpectralForcing fc(global, p);

    const Real dt = 0.01;
    const int  K  = 12;
    for (int s = 0; s < K; ++s) {
        fc.evolve_ou(dt);
        fc.apply(U, local, dt, d.comm());
    }

    // Gather distributed final state onto rank 0.
    std::vector<std::vector<Real>> distributed(NCONS);
    for (int v = 0; v < NCONS; ++v) gather_field(U, v, global, d, distributed[v]);

    int rc = 0;
    if (d.rank() == 0) {
        // Serial reference: same IC, same forcing, run on the global grid.
        State Uref(N, N, N);
        IdealGas eos{GammaLaw{}};
        Grid ref = global;
        for (int k = 0; k < N; ++k)
            for (int j = 0; j < N; ++j)
                for (int i = 0; i < N; ++i) {
                    const Real x = global.x0 + (i + 0.5) * global.dx();
                    const Real y = global.y0 + (j + 0.5) * global.dy();
                    const Real z = global.z0 + (k + 0.5) * global.dz();
                    const Real u = 0.1 * std::sin(x) * std::cos(y);
                    const Real v = 0.1 * std::cos(x) * std::sin(z);
                    const Real w = 0.1 * std::sin(y) * std::sin(z);
                    Uref[RHO](i, j, k)  = 1.0;
                    Uref[RHOU](i, j, k) = u;
                    Uref[RHOV](i, j, k) = v;
                    Uref[RHOW](i, j, k) = w;
                    const Real ke = 0.5 * (u*u + v*v + w*w);
                    Uref[RHOE](i, j, k) = 1.0 / 0.4 + ke;
                }
        BCSet bc2 = bc;
        apply_bcs(Uref, bc2);
        SpectralForcing fcref(global, p);
        for (int s = 0; s < K; ++s) {
            fcref.evolve_ou(dt);
            fcref.apply(Uref, global, dt);
        }

        Real max_diff = 0.0;
        for (int v = 0; v < NCONS; ++v) {
            for (int k = 0; k < N; ++k)
                for (int j = 0; j < N; ++j)
                    for (int i = 0; i < N; ++i) {
                        const std::size_t gi = static_cast<std::size_t>(i)
                            + N * (static_cast<std::size_t>(j) + N * k);
                        const Real d2 = std::fabs(
                            distributed[v][gi] - Uref[v](i, j, k));
                        max_diff = std::max(max_diff, d2);
                    }
        }
        std::printf("[rank0] mpi-forcing K=%d ranks=%d:  max|du| = %.3e\n",
                    K, d.size(), max_diff);
        if (max_diff > 1e-12) {
            std::fprintf(stderr, "[FAIL] forcing not bit-exact: %.3e > 1e-12\n",
                         max_diff);
            rc = 1;
        } else {
            std::printf("[OK] forcing matches serial to round-off\n");
        }
    }

    MPI_Allreduce(&rc, &rc_all, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return rc_all;
}
