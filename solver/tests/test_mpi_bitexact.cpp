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
#include "physics/MixtureEOS.hpp"
#include "physics/Multifluid.hpp"

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

    // Use ny = nz = 16 so a (2,2,1) decomp at 4 ranks gives each rank
    // local_ny = 8 >= NGHOST = 6. Smaller cross-sections cause an
    // unrelated halo-shape bug (TODO: assert this in Domain).
    Grid global_g;
    global_g.nx = 64; global_g.ny = 16; global_g.nz = 16;
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

// Two-gamma multifluid blast: exercises the G-field halo exchange, MPI
// mf_advect_G, gated double-flux contact handling, and the local-gamma dt /
// positivity floor across partitions. Slip-wall box, contact centered at the
// origin (straddles rank boundaries under a 2x2x1 decomp), so internal-face G
// ghosts carry products vs air -- the key thing the halo must get right.
int run_multifluid_bitexact(int K_steps, bool conservative) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Grid global_g;
    global_g.nx = global_g.ny = global_g.nz = 32;
    global_g.lx = global_g.ly = global_g.lz = 1.2;
    global_g.x0 = global_g.y0 = global_g.z0 = -0.6;

    BCSet bc;
    bc.xlo = bc.xhi = BCType::SlipWall;
    bc.ylo = bc.yhi = BCType::SlipWall;
    bc.zlo = bc.zhi = BCType::SlipWall;

    IdealGas eos{GammaLaw{}};            // gamma = 1.4 = gamma_air
    ViscousParams vp; vp.mu = 0.0;
    vp.mf_conservative = conservative;
    const Real Ga = 1.0 / (eos.eos.gamma - 1.0);

    MultifluidParams mp;
    mp.enabled = true; mp.gamma_air = eos.eos.gamma; mp.gamma_p = 1.25;
    mp.R = eos.eos.R; mp.rho_p = 10.0; mp.T_p = 100.0; mp.rho_a = 1.0; mp.T_a = 1.0;
    mp.r0 = 0.1; mp.tanh_thickness = 0.04; mp.Y42_amp = 0.2;
    const Real cfl = 0.3;

    // ---- MPI path ----
    Domain dom(MPI_COMM_WORLD, global_g, bc);
    Grid local_g = dom.local_grid(global_g);
    State U_mpi(local_g.nx, local_g.ny, local_g.nz);
    Field3D G_mpi(local_g.nx, local_g.ny, local_g.nz);
    G_mpi.fill(Ga);
    mf_init_blast(U_mpi, G_mpi, local_g, mp);
    Halo halo(U_mpi, dom);
    halo.exchange(U_mpi);  apply_bcs(U_mpi, bc, dom);
    halo.exchange(G_mpi);  apply_bcs(G_mpi, bc, dom);
    RK3 driver_mpi(local_g.nx, local_g.ny, local_g.nz, U_mpi.ng());
    for (int s = 0; s < K_steps; ++s) {
        Real dt = max_dt_hyperbolic(U_mpi, local_g, eos, cfl, dom.comm(), &G_mpi);
        driver_mpi.step_mpi(U_mpi, local_g, bc, eos, vp, dt, dom, halo, &G_mpi);
        mf_advect_G(G_mpi, U_mpi, local_g, bc, dt, dom, halo);
    }
    std::vector<double> grho, gmom, ge, gG;
    gather_field_to_rank0(U_mpi[RHO ], global_g, dom, grho);
    gather_field_to_rank0(U_mpi[RHOU], global_g, dom, gmom);
    gather_field_to_rank0(U_mpi[RHOE], global_g, dom, ge);
    gather_field_to_rank0(G_mpi,       global_g, dom, gG);

    int fail = 0;
    if (rank == 0) {
        State U_ser(global_g.nx, global_g.ny, global_g.nz);
        Field3D G_ser(global_g.nx, global_g.ny, global_g.nz);
        G_ser.fill(Ga);
        mf_init_blast(U_ser, G_ser, global_g, mp);
        mf_fill_G_bcs(G_ser, bc);
        auto total_E = [&](const State& U) {
            double s = 0.0;
            for (int k = 0; k < global_g.nz; ++k)
                for (int j = 0; j < global_g.ny; ++j)
                    for (int i = 0; i < global_g.nx; ++i) s += U[RHOE](i, j, k);
            return s;
        };
        const double E0 = total_E(U_ser);
        RK3 driver_ser(global_g.nx, global_g.ny, global_g.nz, U_ser.ng());
        for (int s = 0; s < K_steps; ++s) {
            Real dt = max_dt_hyperbolic(U_ser, global_g, eos, cfl, &G_ser);
            driver_ser.step(U_ser, global_g, bc, eos, vp, dt, &G_ser);
            mf_advect_G(G_ser, U_ser, global_g, bc, dt);
        }
        const Real er = max_abs_error(grho, U_ser[RHO ], global_g);
        const Real em = max_abs_error(gmom, U_ser[RHOU], global_g);
        const Real ee = max_abs_error(ge,   U_ser[RHOE], global_g);
        const Real eg = max_abs_error(gG,   G_ser,       global_g);
        const Real tol = 1e-12;
        const double dErel = std::fabs(total_E(U_ser) - E0) / std::fabs(E0);
        const bool bitexact = (er < tol && em < tol && ee < tol && eg < tol);
        // Conservative mode must conserve total energy to ~round-off; the
        // double-flux mode is expected to drift (we only report it there).
        const bool cons_ok = !conservative || dErel < 1e-10;
        std::printf("Multifluid blast (%s), K=%d steps, ranks=%d: "
                    "max|dRho|=%.2e max|dMom|=%.2e max|dE|=%.2e max|dG|=%.2e "
                    "dE_tot/E0=%.2e %s\n",
                    conservative ? "conservative" : "double-flux",
                    K_steps, size, er, em, ee, eg, dErel,
                    (bitexact && cons_ok) ? "PASS" : "FAIL");
        if (!bitexact || !cons_ok) fail = 1;
    }
    MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return fail;
}

// JWL multifluid blast: same machinery as the two-gamma case but with the JWL
// products EOS selected by a phi marker. Exercises the phi halo exchange, MPI
// mf_advect_G, JWL frozen-EOS double-flux, and the JWL local sound speed / dt
// across partitions. Serial and MPI use the identical MixtureEOS, so any
// mismatch isolates the domain decomposition.
int run_jwl_bitexact(int K_steps) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Grid global_g;
    global_g.nx = global_g.ny = global_g.nz = 32;
    global_g.lx = global_g.ly = global_g.lz = 1.0;
    global_g.x0 = global_g.y0 = global_g.z0 = -0.5;

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Outflow;
    bc.ylo = bc.yhi = BCType::Outflow;
    bc.zlo = bc.zhi = BCType::Outflow;

    IdealGas eos{GammaLaw{}};            // gamma = 1.4 = air
    ViscousParams vp; vp.mu = 0.0;

    // Nondimensional TNT (rho_ref=1630, p_ref=21e9): products O(1), air ~ 1e-3.
    const Real rr = 1630.0, pr = 21.0e9;
    MultifluidParams mp;
    mp.enabled = true; mp.gamma_air = eos.eos.gamma; mp.R = eos.eos.R;
    mp.jwl_mode = true; mp.phi_switch = 0.5;
    mp.jwl.A = 3.712e11 / pr; mp.jwl.B = 3.231e9 / pr;
    mp.jwl.R1 = 4.15; mp.jwl.R2 = 0.95; mp.jwl.omega = 0.30;
    mp.jwl.rho0 = 1630.0 / rr; mp.jwl.E0 = 7.0e9 / pr;
    mp.rho_cj = 2228.0 / rr; mp.p_cj = 21.0e9 / pr;
    mp.rho_a = 1.2 / rr; mp.p_a = 1.013e5 / pr;
    mp.r0 = 0.12; mp.tanh_thickness = 0.04; mp.Y42_amp = 0.2;
    vp.rho_floor  = 1e-3 * mp.rho_a;
    vp.eint_floor = 1e-3 * (mp.p_a / (mp.gamma_air - 1.0));
    const MixtureEOS mix = mp.mixture();
    const Real cfl = 0.3;

    // ---- MPI path ----
    Domain dom(MPI_COMM_WORLD, global_g, bc);
    Grid local_g = dom.local_grid(global_g);
    State U_mpi(local_g.nx, local_g.ny, local_g.nz);
    Field3D G_mpi(local_g.nx, local_g.ny, local_g.nz);
    G_mpi.fill(0.0);
    mf_init_blast(U_mpi, G_mpi, local_g, mp);
    Halo halo(U_mpi, dom);
    halo.exchange(U_mpi);  apply_bcs(U_mpi, bc, dom);
    halo.exchange(G_mpi);  apply_bcs(G_mpi, bc, dom);
    RK3 driver_mpi(local_g.nx, local_g.ny, local_g.nz, U_mpi.ng());
    for (int s = 0; s < K_steps; ++s) {
        Real dt = max_dt_hyperbolic(U_mpi, local_g, eos, cfl, dom.comm(), &G_mpi, &mix);
        driver_mpi.step_mpi(U_mpi, local_g, bc, eos, vp, dt, dom, halo, &G_mpi, &mix);
        mf_advect_G(G_mpi, U_mpi, local_g, bc, dt, dom, halo);
    }
    std::vector<double> grho, gmom, ge, gG;
    gather_field_to_rank0(U_mpi[RHO ], global_g, dom, grho);
    gather_field_to_rank0(U_mpi[RHOU], global_g, dom, gmom);
    gather_field_to_rank0(U_mpi[RHOE], global_g, dom, ge);
    gather_field_to_rank0(G_mpi,       global_g, dom, gG);

    int fail = 0;
    if (rank == 0) {
        State U_ser(global_g.nx, global_g.ny, global_g.nz);
        Field3D G_ser(global_g.nx, global_g.ny, global_g.nz);
        G_ser.fill(0.0);
        mf_init_blast(U_ser, G_ser, global_g, mp);
        mf_fill_G_bcs(G_ser, bc);
        RK3 driver_ser(global_g.nx, global_g.ny, global_g.nz, U_ser.ng());
        for (int s = 0; s < K_steps; ++s) {
            Real dt = max_dt_hyperbolic(U_ser, global_g, eos, cfl, &G_ser, &mix);
            driver_ser.step(U_ser, global_g, bc, eos, vp, dt, &G_ser, &mix);
            mf_advect_G(G_ser, U_ser, global_g, bc, dt);
        }
        const Real er = max_abs_error(grho, U_ser[RHO ], global_g);
        const Real em = max_abs_error(gmom, U_ser[RHOU], global_g);
        const Real ee = max_abs_error(ge,   U_ser[RHOE], global_g);
        const Real eg = max_abs_error(gG,   G_ser,       global_g);
        const Real tol = 1e-12;
        const bool bitexact = (er < tol && em < tol && ee < tol && eg < tol);
        std::printf("JWL blast, K=%d steps, ranks=%d: "
                    "max|dRho|=%.2e max|dMom|=%.2e max|dE|=%.2e max|dPhi|=%.2e %s\n",
                    K_steps, size, er, em, ee, eg, bitexact ? "PASS" : "FAIL");
        if (!bitexact) fail = 1;
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
        total_fail += run_multifluid_bitexact(15, /*conservative=*/false);
        total_fail += run_multifluid_bitexact(15, /*conservative=*/true);
        total_fail += run_jwl_bitexact(15);
    }

    if (rank == 0)
        std::printf("OVERALL bit-exact: %s (total_fail=%d)\n",
                    total_fail == 0 ? "PASS" : "FAIL", total_fail);
    MPI_Finalize();
    return total_fail;
}
