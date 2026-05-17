// MPI halo exchange test. Run via mpirun -n {1, 2, 4, 8}.
//
// Fill the GLOBAL conserved-state space with a smooth analytic pattern using
// each rank's local cell coordinates. After Halo::exchange + apply_bcs
// (slip-wall + periodic), every ghost cell should hold the value of the
// analytic pattern at that ghost cell's global coordinate. Verify cell-by-
// cell at chosen probe points just inside the ghost layer.

#include "bc/BC.hpp"
#include "core/Config.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "ic/Canonical.hpp"
#include "parallel/Domain.hpp"
#include "parallel/Halo.hpp"
#include "physics/EOS.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace blast;

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int fail_count = 0;

    // ============================================================
    // Test 1: periodic in all 3 dims; smooth pattern reads back
    // exactly via halo exchange.
    // ============================================================
    {
        Grid g_global;
        g_global.nx = g_global.ny = g_global.nz = 32;
        g_global.lx = g_global.ly = g_global.lz = 2.0 * M_PI;
        g_global.x0 = g_global.y0 = g_global.z0 = 0.0;

        BCSet bc;
        bc.xlo = bc.xhi = BCType::Periodic;
        bc.ylo = bc.yhi = BCType::Periodic;
        bc.zlo = bc.zhi = BCType::Periodic;

        Domain dom(MPI_COMM_WORLD, g_global, bc);
        Grid g_local = dom.local_grid(g_global);

        IdealGas eos{GammaLaw{1.4, 1.0}};
        State U(g_local.nx, g_local.ny, g_local.nz);

        // Fill local interior with rho(x,y,z) = 2 + sin(x)*sin(y)*sin(z),
        // velocities zero, p uniform.
        auto rho_at = [](Real x, Real y, Real z) {
            return 2.0 + std::sin(x) * std::sin(y) * std::sin(z);
        };
        for (int k = 0; k < g_local.nz; ++k)
            for (int j = 0; j < g_local.ny; ++j)
                for (int i = 0; i < g_local.nx; ++i) {
                    set_from_primitive(U, i, j, k, eos,
                                       rho_at(g_local.xc(i), g_local.yc(j), g_local.zc(k)),
                                       0, 0, 0, 1.0);
                }

        Halo halo(U, dom);
        halo.exchange(U);
        apply_bcs(U, bc, dom);

        // Probe each ghost layer just inside [-1, n_local] -- the layer
        // closest to the interior. Check value matches analytic at that
        // ghost cell's GLOBAL coordinate (since the pattern is periodic
        // with period 2 pi = L, sin(x + L) = sin(x), so the analytic
        // function repeats and the cell's local-coordinate version is
        // also valid).
        int local_fail = 0;
        const int ng = U.ng();
        for (int d = 0; d < 3; ++d) {
            for (int side : {-1, +1}) {
                // Probe at the first ghost cell on this face.
                int gi = (d == 0) ? (side < 0 ? -1 : g_local.nx) : g_local.nx / 2;
                int gj = (d == 1) ? (side < 0 ? -1 : g_local.ny) : g_local.ny / 2;
                int gk = (d == 2) ? (side < 0 ? -1 : g_local.nz) : g_local.nz / 2;

                const Real x = g_local.xc(gi);
                const Real y = g_local.yc(gj);
                const Real z = g_local.zc(gk);
                const Real expected = rho_at(x, y, z);
                const Real got = U[RHO](gi, gj, gk);
                if (std::fabs(got - expected) > 1e-12) {
                    ++local_fail;
                    if (world_rank == 0)
                        std::fprintf(stderr,
                            "FAIL[rank=%d, halo periodic] face d=%d side=%d "
                            "at (%d,%d,%d): got=%.16f expected=%.16f\n",
                            world_rank, d, side, gi, gj, gk, got, expected);
                }
            }
        }
        int global_fail = 0;
        MPI_Allreduce(&local_fail, &global_fail, 1, MPI_INT, MPI_SUM,
                      MPI_COMM_WORLD);
        if (world_rank == 0)
            std::printf("test1 periodic halo: %s (failures=%d)\n",
                        global_fail == 0 ? "PASS" : "FAIL", global_fail);
        fail_count += global_fail;
        (void)ng;
    }

    // ============================================================
    // Test 2: physical-face count. Each wall is split across the ranks
    // that border it. Total physical faces across ranks should equal
    //   2 * (Npy*Npz + Npx*Npz + Npx*Npy)
    // where (Npx, Npy, Npz) = dom.dims().
    // ============================================================
    {
        Grid g_global;
        g_global.nx = g_global.ny = g_global.nz = 16;
        g_global.lx = g_global.ly = g_global.lz = 1.0;
        g_global.x0 = g_global.y0 = g_global.z0 = 0.0;

        BCSet bc;
        bc.xlo = bc.xhi = BCType::SlipWall;
        bc.ylo = bc.yhi = BCType::SlipWall;
        bc.zlo = bc.zhi = BCType::SlipWall;

        Domain dom(MPI_COMM_WORLD, g_global, bc);

        int local_phys = 0;
        for (int d = 0; d < 3; ++d)
            for (int side : {-1, +1})
                if (dom.is_physical_face(d, side)) ++local_phys;

        int global_phys = 0;
        MPI_Allreduce(&local_phys, &global_phys, 1, MPI_INT, MPI_SUM,
                      MPI_COMM_WORLD);

        const auto& nd = dom.dims();
        const int expected =
            2 * (nd[1] * nd[2] + nd[0] * nd[2] + nd[0] * nd[1]);
        if (world_rank == 0)
            std::printf("test2 slip-wall face count: %s (got=%d expected=%d)\n",
                        global_phys == expected ? "PASS" : "FAIL",
                        global_phys, expected);
        if (global_phys != expected) ++fail_count;
    }

    // ============================================================
    // Test 3: global cell extents sum across ranks to the global N.
    // ============================================================
    {
        Grid g_global;
        g_global.nx = 33; g_global.ny = 17; g_global.nz = 9;  // non-divisible
        g_global.lx = g_global.ly = g_global.lz = 1.0;
        g_global.x0 = g_global.y0 = g_global.z0 = 0.0;
        BCSet bc;
        bc.xlo = bc.xhi = BCType::Periodic;
        bc.ylo = bc.yhi = BCType::Periodic;
        bc.zlo = bc.zhi = BCType::Periodic;
        Domain dom(MPI_COMM_WORLD, g_global, bc);
        Grid lg = dom.local_grid(g_global);

        long long local_N = static_cast<long long>(lg.nx) * lg.ny * lg.nz;
        long long total = 0;
        MPI_Allreduce(&local_N, &total, 1, MPI_LONG_LONG, MPI_SUM,
                      MPI_COMM_WORLD);
        long long expected = static_cast<long long>(g_global.nx) * g_global.ny * g_global.nz;
        if (world_rank == 0)
            std::printf("test3 cell-count sum: %s (got=%lld expected=%lld)\n",
                        total == expected ? "PASS" : "FAIL", total, expected);
        if (total != expected) ++fail_count;
    }

    if (world_rank == 0)
        std::printf("OVERALL: %s (fail_count=%d, ranks=%d)\n",
                    fail_count == 0 ? "PASS" : "FAIL", fail_count, world_size);

    MPI_Finalize();
    return fail_count;
}
