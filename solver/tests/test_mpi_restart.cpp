// MPI restart round-trip. Each rank fills its local interior with a smooth
// analytic pattern parameterized by global coordinates, calls the MPI
// write_checkpoint(...) collectively, then a fresh State is built and the
// checkpoint is read back. Both passes use the same Domain so the global
// shape is preserved. Verify cell-by-cell that the readback matches the
// original on every rank.

#include "core/Config.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "ic/Canonical.hpp"
#include "io/Restart.hpp"
#include "parallel/Domain.hpp"
#include "physics/EOS.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace blast;

namespace {

void fill_pattern(State& U, const Grid& g, const IdealGas& eos) {
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real x = g.xc(i), y = g.yc(j), z = g.zc(k);
                const Real rho = 1.0 + 0.1 * std::sin(x) * std::cos(y) * std::sin(z);
                const Real u   = 0.2 * std::cos(x) * std::sin(y);
                const Real v   = 0.3 * std::sin(2.0 * x);
                const Real w   = 0.05 * std::cos(y + z);
                const Real p   = 1.0 + 0.05 * std::cos(x + y + z);
                set_from_primitive(U, i, j, k, eos, rho, u, v, w, p);
            }
}

Real max_abs_diff(const State& A, const State& B, int n_vars) {
    Real m = 0.0;
    for (int v = 0; v < n_vars; ++v)
        for (int k = 0; k < A.nz(); ++k)
            for (int j = 0; j < A.ny(); ++j)
                for (int i = 0; i < A.nx(); ++i) {
                    m = std::max(m, std::fabs(A[v](i, j, k) - B[v](i, j, k)));
                }
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int fail = 0;
    {
        Grid global_g;
        global_g.nx = global_g.ny = global_g.nz = 16;
        global_g.lx = global_g.ly = global_g.lz = 2.0 * M_PI;
        global_g.x0 = global_g.y0 = global_g.z0 = 0.0;
        BCSet bc;
        bc.xlo = bc.xhi = BCType::Periodic;
        bc.ylo = bc.yhi = BCType::Periodic;
        bc.zlo = bc.zhi = BCType::Periodic;
        IdealGas eos{GammaLaw{}};

        Domain dom(MPI_COMM_WORLD, global_g, bc);
        Grid local_g = dom.local_grid(global_g);

        State A(local_g.nx, local_g.ny, local_g.nz);
        fill_pattern(A, local_g, eos);

        // Per-rank temp path so we don't collide. Actually the checkpoint
        // must be a SINGLE shared file -- use a fixed path under /tmp.
        const std::string path = "/tmp/blast_mpi_ckpt_test.ckpt.h5";
        write_checkpoint(path, A, global_g, /*t=*/0.0125, /*step=*/77, dom);

        State B(local_g.nx, local_g.ny, local_g.nz);
        auto h = read_checkpoint(path, B, global_g, dom);

        if (h.time != 0.0125 || h.step != 77 ||
            h.nx != global_g.nx || h.ny != global_g.ny || h.nz != global_g.nz) {
            ++fail;
            if (rank == 0)
                std::fprintf(stderr,
                    "FAIL header: time=%g step=%d shape=(%d,%d,%d)\n",
                    h.time, h.step, h.nx, h.ny, h.nz);
        }

        const Real local_err = max_abs_diff(A, B, NCONS);
        Real global_err = 0.0;
        MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);
        if (global_err > 0.0) {
            ++fail;
            if (rank == 0)
                std::fprintf(stderr,
                    "FAIL state: max|delta| = %.3e at ranks=%d\n",
                    global_err, size);
        }

        if (rank == 0)
            std::printf("MPI restart round-trip at %d ranks: "
                        "max|delta|=%.3e header.time=%g header.step=%d %s\n",
                        size, global_err, h.time, h.step,
                        (fail == 0) ? "PASS" : "FAIL");

        // Rank 0 cleans up the temp file.
        if (rank == 0) std::remove(path.c_str());
    }

    MPI_Finalize();
    return fail;
}
