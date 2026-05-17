// MPI bit-exactness regression: run a canonical problem under MPI on N
// ranks, gather the result to rank 0, then run the same problem serially
// on rank 0 and verify the state matches cell-by-cell. Catches regressions
// in halo exchange, BC dispatch on physical faces, dt reduction, and
// flux-difference stencils across partitions.

#include "bc/BC.hpp"
#include "core/Config.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "ic/Canonical.hpp"
#include "numerics/RHS.hpp"
#include "numerics/RK3.hpp"
#include "parallel/Domain.hpp"
#include "parallel/Halo.hpp"
#include "physics/EOS.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace blast;

namespace {

// Sod 1D IC filled across the FULL padded region (interior + ghosts) using
// an explicit global x-midpoint. Avoids ic_sod_x's internal use of the
// local grid center.
void fill_sod_explicit(State& U, const Grid& g, const IdealGas& eos,
                       Real x_mid) {
    for (int k = -U.ng(); k < g.nz + U.ng(); ++k)
        for (int j = -U.ng(); j < g.ny + U.ng(); ++j)
            for (int i = -U.ng(); i < g.nx + U.ng(); ++i) {
                const Real x = g.xc(i);
                const Real rho = (x < x_mid) ? 1.0 : 0.125;
                const Real p   = (x < x_mid) ? 1.0 : 0.1;
                set_from_primitive(U, i, j, k, eos, rho, 0.0, 0.0, 0.0, p);
            }
}

// Sedov-3D explicit blast (uses global center & global volume).
void fill_sedov_explicit(State& U, const Grid& g, const IdealGas& eos,
                         Real E_total, Real rho_amb, Real p_amb,
                         Real r_blast, Real xc, Real yc, Real zc) {
    const Real V_blast = (4.0 / 3.0) * M_PI * r_blast * r_blast * r_blast;
    const Real p_blast = (eos.eos.gamma - 1.0) * E_total / V_blast;
    for (int k = -U.ng(); k < g.nz + U.ng(); ++k)
        for (int j = -U.ng(); j < g.ny + U.ng(); ++j)
            for (int i = -U.ng(); i < g.nx + U.ng(); ++i) {
                const Real dx = g.xc(i) - xc;
                const Real dy = g.yc(j) - yc;
                const Real dz = g.zc(k) - zc;
                const Real r  = std::sqrt(dx*dx + dy*dy + dz*dz);
                const Real p  = (r < r_blast) ? p_blast : p_amb;
                set_from_primitive(U, i, j, k, eos, rho_amb, 0.0, 0.0, 0.0, p);
            }
}

void gather_field_to_rank0(const Field3D& f, const Grid& global_g,
                           const Domain& d, std::vector<double>& out) {
    const int rank = d.rank(), size = d.size();
    const auto ext = d.global_extent();
    const int nx_g = ext[0], ny_g = ext[1], nz_g = ext[2];
    const int nx_l = f.nx(), ny_l = f.ny(), nz_l = f.nz();

    std::vector<double> local(static_cast<std::size_t>(nx_l) * ny_l * nz_l);
    for (int k = 0; k < nz_l; ++k)
        for (int j = 0; j < ny_l; ++j)
            for (int i = 0; i < nx_l; ++i) {
                const std::size_t li = static_cast<std::size_t>(i)
                    + nx_l * (static_cast<std::size_t>(j) + ny_l * k);
                local[li] = f(i, j, k);
            }

    if (rank == 0) {
        out.assign(static_cast<std::size_t>(nx_g) * ny_g * nz_g, 0.0);
        const auto off0 = d.global_offset(global_g);
        for (int k = 0; k < nz_l; ++k)
            for (int j = 0; j < ny_l; ++j)
                for (int i = 0; i < nx_l; ++i) {
                    const std::size_t li = static_cast<std::size_t>(i)
                        + nx_l * (static_cast<std::size_t>(j) + ny_l * k);
                    const std::size_t gi = static_cast<std::size_t>(i + off0[0])
                        + nx_g * (static_cast<std::size_t>(j + off0[1])
                                  + ny_g * (k + off0[2]));
                    out[gi] = local[li];
                }
        for (int r = 1; r < size; ++r) {
            int meta[6];
            MPI_Recv(meta, 6, MPI_INT, r, 9000, d.comm(), MPI_STATUS_IGNORE);
            const int ox = meta[0], oy = meta[1], oz = meta[2];
            const int ex = meta[3], ey = meta[4], ez = meta[5];
            std::vector<double> rbuf(static_cast<std::size_t>(ex) * ey * ez);
            MPI_Recv(rbuf.data(), static_cast<int>(rbuf.size()), MPI_DOUBLE, r,
                     9001, d.comm(), MPI_STATUS_IGNORE);
            for (int k = 0; k < ez; ++k)
                for (int j = 0; j < ey; ++j)
                    for (int i = 0; i < ex; ++i) {
                        const std::size_t li = static_cast<std::size_t>(i)
                            + ex * (static_cast<std::size_t>(j) + ey * k);
                        const std::size_t gi = static_cast<std::size_t>(i + ox)
                            + nx_g * (static_cast<std::size_t>(j + oy)
                                      + ny_g * (k + oz));
                        out[gi] = rbuf[li];
                    }
        }
    } else {
        const auto off = d.global_offset(global_g);
        int meta[6] = {static_cast<int>(off[0]), static_cast<int>(off[1]),
                       static_cast<int>(off[2]), nx_l, ny_l, nz_l};
        MPI_Send(meta, 6, MPI_INT, 0, 9000, d.comm());
        MPI_Send(local.data(), static_cast<int>(local.size()),
                 MPI_DOUBLE, 0, 9001, d.comm());
    }
}

// Compare a gathered MPI field against a serial reference field on rank 0.
// Returns max abs error.
Real max_abs_error(const std::vector<double>& mpi_buf,
                   const Field3D& serial,
                   const Grid& g) {
    const int nx = g.nx, ny = g.ny, nz = g.nz;
    Real m = 0.0;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const std::size_t gi = static_cast<std::size_t>(i)
                    + nx * (static_cast<std::size_t>(j) + ny * k);
                m = std::max(m, std::fabs(mpi_buf[gi] - serial(i, j, k)));
            }
    return m;
}

// Returns 0 on PASS, 1 on FAIL.
int run_sod_bitexact(int K_steps) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Grid global_g;
    global_g.nx = 64; global_g.ny = 4; global_g.nz = 4;
    global_g.lx = 1.0;
    global_g.ly = global_g.lx / global_g.nx * global_g.ny;
    global_g.lz = global_g.ly;
    global_g.x0 = 0.0; global_g.y0 = 0.0; global_g.z0 = 0.0;
    const Real x_mid = 0.5;

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Outflow;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;

    IdealGas eos{GammaLaw{}};
    ViscousParams vp; vp.mu = 0.0;

    // ---- MPI path ----
    Domain dom(MPI_COMM_WORLD, global_g, bc);
    Grid local_g = dom.local_grid(global_g);
    State U_mpi(local_g.nx, local_g.ny, local_g.nz);
    fill_sod_explicit(U_mpi, local_g, eos, x_mid);

    Halo halo(U_mpi, dom);
    RK3 driver_mpi(local_g.nx, local_g.ny, local_g.nz, U_mpi.ng());
    for (int s = 0; s < K_steps; ++s) {
        Real dt = max_dt_hyperbolic(U_mpi, local_g, eos, 0.4, dom.comm());
        driver_mpi.step_mpi(U_mpi, local_g, bc, eos, vp, dt, dom, halo);
    }

    std::vector<double> grho, gmom, ge;
    gather_field_to_rank0(U_mpi[RHO ], global_g, dom, grho);
    gather_field_to_rank0(U_mpi[RHOU], global_g, dom, gmom);
    gather_field_to_rank0(U_mpi[RHOE], global_g, dom, ge);

    int fail = 0;
    if (rank == 0) {
        State U_ser(global_g.nx, global_g.ny, global_g.nz);
        fill_sod_explicit(U_ser, global_g, eos, x_mid);
        RK3 driver_ser(global_g.nx, global_g.ny, global_g.nz, U_ser.ng());
        for (int s = 0; s < K_steps; ++s) {
            Real dt = max_dt_hyperbolic(U_ser, global_g, eos, 0.4);
            driver_ser.step(U_ser, global_g, bc, eos, vp, dt);
        }
        const Real er = max_abs_error(grho, U_ser[RHO ], global_g);
        const Real em = max_abs_error(gmom, U_ser[RHOU], global_g);
        const Real ee = max_abs_error(ge,   U_ser[RHOE], global_g);
        const Real tol = 1e-12;
        std::printf("Sod 1D, K=%d steps, ranks=%d: "
                    "max|dRho|=%.2e max|dMom|=%.2e max|dE|=%.2e %s\n",
                    K_steps, size, er, em, ee,
                    (er < tol && em < tol && ee < tol) ? "PASS" : "FAIL");
        if (er > tol || em > tol || ee > tol) fail = 1;
    }
    MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return fail;
}

int run_sedov_bitexact(int K_steps) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Grid global_g;
    global_g.nx = global_g.ny = global_g.nz = 32;
    global_g.lx = global_g.ly = global_g.lz = 1.2;
    global_g.x0 = global_g.y0 = global_g.z0 = -0.6;

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Outflow;
    bc.ylo = bc.yhi = BCType::Outflow;
    bc.zlo = bc.zhi = BCType::Outflow;

    IdealGas eos{GammaLaw{}};
    ViscousParams vp; vp.mu = 0.0;

    const Real E_total = 1.0, rho_amb = 1.0, p_amb = 1e-5;
    const Real r_blast = 4.0 * global_g.dx();
    const Real xc = 0.0, yc = 0.0, zc = 0.0;

    Domain dom(MPI_COMM_WORLD, global_g, bc);
    Grid local_g = dom.local_grid(global_g);
    State U_mpi(local_g.nx, local_g.ny, local_g.nz);
    fill_sedov_explicit(U_mpi, local_g, eos, E_total, rho_amb, p_amb,
                        r_blast, xc, yc, zc);
    Halo halo(U_mpi, dom);
    RK3 driver_mpi(local_g.nx, local_g.ny, local_g.nz, U_mpi.ng());
    for (int s = 0; s < K_steps; ++s) {
        Real dt = max_dt_hyperbolic(U_mpi, local_g, eos, 0.3, dom.comm());
        driver_mpi.step_mpi(U_mpi, local_g, bc, eos, vp, dt, dom, halo);
    }
    std::vector<double> grho;
    gather_field_to_rank0(U_mpi[RHO], global_g, dom, grho);

    int fail = 0;
    if (rank == 0) {
        State U_ser(global_g.nx, global_g.ny, global_g.nz);
        fill_sedov_explicit(U_ser, global_g, eos, E_total, rho_amb, p_amb,
                            r_blast, xc, yc, zc);
        RK3 driver_ser(global_g.nx, global_g.ny, global_g.nz, U_ser.ng());
        for (int s = 0; s < K_steps; ++s) {
            Real dt = max_dt_hyperbolic(U_ser, global_g, eos, 0.3);
            driver_ser.step(U_ser, global_g, bc, eos, vp, dt);
        }
        const Real er = max_abs_error(grho, U_ser[RHO], global_g);
        const Real tol = 1e-12;
        std::printf("Sedov 3D, K=%d steps, ranks=%d: max|dRho|=%.2e %s\n",
                    K_steps, size, er, er < tol ? "PASS" : "FAIL");
        if (er > tol) fail = 1;
    }
    MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return fail;
}

}  // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int total_fail = 0;
    {
        total_fail += run_sod_bitexact(20);
        total_fail += run_sedov_bitexact(15);
    }

    if (rank == 0)
        std::printf("OVERALL bit-exact: %s (total_fail=%d)\n",
                    total_fail == 0 ? "PASS" : "FAIL", total_fail);
    MPI_Finalize();
    return total_fail;
}
