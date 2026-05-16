#include <gtest/gtest.h>

#include "bc/BC.hpp"
#include "core/Config.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "diagnostics/Statistics.hpp"
#include "ic/Canonical.hpp"
#include "numerics/RHS.hpp"
#include "numerics/RK3.hpp"
#include "physics/EOS.hpp"

#include <cmath>

using namespace blast;

TEST(ChamberSmoke, BlastSettlesWithSlipWalls) {
    Grid g;
    g.nx = g.ny = g.nz = 32;
    g.lx = g.ly = g.lz = 1.0;
    g.x0 = g.y0 = g.z0 = -0.5;

    // Use R = 1, T_ambient = 1 so p_ambient = R rho T = 1.
    GammaLaw gl; gl.gamma = 1.4; gl.R = 1.0;
    IdealGas eos{gl};
    State U(g.nx, g.ny, g.nz);

    const Real rho_blast    = 10.0;
    const Real T_blast      = 100.0;
    const Real rho_ambient  = 1.0;
    const Real T_ambient    = 1.0;
    const Real r_blast      = 0.1;
    const Real thickness    = 1.5 * g.dx();
    const Real Y42_amp      = 0.0;

    ic_sphere_blast_3d(U, g, eos, rho_blast, T_blast,
                       rho_ambient, T_ambient, r_blast, thickness, Y42_amp);

    BCSet bc;
    bc.xlo = bc.xhi = BCType::SlipWall;
    bc.ylo = bc.yhi = BCType::SlipWall;
    bc.zlo = bc.zhi = BCType::SlipWall;

    RK3 driver(g.nx, g.ny, g.nz, U.ng());

    // Initial total energy in the box (sum of conserved E density).
    Real E0 = 0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
                E0 += U[RHOE](i, j, k);

    Real t = 0.0;
    int steps = 0;
    const int max_steps = 200;
    while (steps < max_steps) {
        Real dt = max_dt_hyperbolic(U, g, eos, 0.3);
        ASSERT_TRUE(std::isfinite(dt) && dt > 0.0) << "non-finite dt at step " << steps;
        driver.step(U, g, bc, eos, dt);
        t += dt;
        ++steps;
    }

    // Total energy: with slip walls (zero mass flux, zero pressure work on
    // walls since u_n = 0) and no viscous dissipation, the inviscid
    // conservative scheme should preserve total energy to numerical roundoff.
    Real E1 = 0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
                E1 += U[RHOE](i, j, k);
    const Real rel = std::fabs(E1 - E0) / std::fabs(E0);
    EXPECT_LT(rel, 1e-3) << "rel energy error " << rel;

    // Kinetic energy was zero at t=0; should be positive after shock release.
    auto s = velocity_stats(U, eos);
    EXPECT_GT(s.ke_total, 0.0);

    // Density and pressure positivity preserved.
    Real rho_min = 1e30, p_min = 1e30;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real r = U[RHO](i, j, k);
                const Real u = U[RHOU](i, j, k) / r;
                const Real v = U[RHOV](i, j, k) / r;
                const Real w = U[RHOW](i, j, k) / r;
                const Real ke = 0.5 * r * (u*u + v*v + w*w);
                const Real p  = eos.pressure(r, U[RHOE](i, j, k) - ke);
                rho_min = std::min(rho_min, r);
                p_min   = std::min(p_min,   p);
            }
    EXPECT_GT(rho_min, 0.0) << "negative density appeared";
    EXPECT_GT(p_min,   0.0) << "negative pressure appeared";
}
