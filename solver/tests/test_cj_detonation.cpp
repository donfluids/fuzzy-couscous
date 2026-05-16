#include <gtest/gtest.h>

#include "core/Config.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "ic/Canonical.hpp"
#include "physics/EOS.hpp"

#include <cmath>

using namespace blast;

TEST(CJDetonation, PostShockStateMatchesAnalytic) {
    // Dimensionless test: rho_0 = 1, T_0 = 1, R = 1 -> p_0 = 1, c_0 = sqrt(gamma).
    // Heat release q chosen large enough to be in the strong-detonation regime.
    GammaLaw gl; gl.gamma = 1.4; gl.R = 1.0;
    IdealGas eos{gl};

    Grid g; g.nx = g.ny = g.nz = 32;
    g.lx = g.ly = g.lz = 1.0;
    g.x0 = g.y0 = g.z0 = -0.5;

    const Real rho_0 = 1.0, T_0 = 1.0;
    const Real q     = 50.0;       // q / c_0^2 = 50/1.4 ~ 36 -> strong limit
    const Real r_cj  = 0.15;

    State U(g.nx, g.ny, g.nz);
    ic_cj_detonation_3d(U, g, eos, rho_0, T_0, q, r_cj, /*thickness=*/0.0,
                        /*Y42=*/0.0);

    // Recompute the analytic CJ state and check that the cell at the origin
    // (which is inside r_cj) carries that state.
    const Real gamma = gl.gamma;
    const Real c_0   = std::sqrt(gamma * gl.R * T_0);
    const Real alpha = (gamma + 1.0) * q / (c_0 * c_0);
    const Real M_D2  = 1.0 + alpha + std::sqrt(alpha * alpha + 2.0 * alpha);
    const Real p_0_ref   = rho_0 * gl.R * T_0;
    const Real p_cj_an   = p_0_ref * (1.0 + gamma * M_D2) / (gamma + 1.0);
    const Real rho_cj_an = rho_0   * (gamma + 1.0) * M_D2 / (gamma * M_D2 + 1.0);

    // Center cell at (16, 16, 16) is inside the sphere; pull primitives.
    const int ic = g.nx / 2, jc = g.ny / 2, kc = g.nz / 2;
    const Real r = U[RHO ](ic, jc, kc);
    const Real u = U[RHOU](ic, jc, kc) / r;
    const Real v = U[RHOV](ic, jc, kc) / r;
    const Real w = U[RHOW](ic, jc, kc) / r;
    const Real ke = 0.5 * r * (u*u + v*v + w*w);
    const Real p = eos.pressure(r, U[RHOE](ic, jc, kc) - ke);

    EXPECT_NEAR(r, rho_cj_an, 1e-6 * rho_cj_an);
    EXPECT_NEAR(p, p_cj_an,   1e-6 * p_cj_an);
    // Note: for even nx there is no cell exactly at the geometric origin,
    // so velocity at (ic, jc, kc) is nonzero (it's a small but real radial
    // velocity at that off-center cell). Don't assert v=0 there.
    (void)u; (void)v; (void)w;

    // A cell well off-center but still inside the sphere should have radial
    // flow matching u_cj (in magnitude).
    const Real u_cj_an = c_0 * std::sqrt(M_D2) * (1.0 - rho_0 / rho_cj_an);
    const int i1 = ic + 3, j1 = jc, k1 = kc;
    const Real r1 = U[RHO ](i1, j1, k1);
    const Real ux = U[RHOU](i1, j1, k1) / r1;
    const Real uy = U[RHOV](i1, j1, k1) / r1;
    const Real uz = U[RHOW](i1, j1, k1) / r1;
    const Real speed = std::sqrt(ux*ux + uy*uy + uz*uz);
    EXPECT_NEAR(speed, u_cj_an, 1e-6 * u_cj_an);
    // The velocity vector must point radially outward.
    const Real x = g.xc(i1) - (g.x0 + 0.5 * g.lx);
    EXPECT_GT(ux * x, 0.0);
}

TEST(CJDetonation, WeakHeatReleaseConvergesToShock) {
    // M_D -> 1 (sonic) and rho_cj/rho_0 -> 1 only when alpha = (gamma+1)q/c^2
    // is truly small. With c_0^2 = gamma R T_0 = 1.4 in our nondimensional
    // units we need q << c_0^2. Use q = 1e-15.
    GammaLaw gl; gl.gamma = 1.4; gl.R = 1.0;
    IdealGas eos{gl};

    Grid g; g.nx = g.ny = g.nz = 16;
    g.lx = g.ly = g.lz = 1.0; g.x0 = g.y0 = g.z0 = -0.5;
    State U(g.nx, g.ny, g.nz);

    ic_cj_detonation_3d(U, g, eos, /*rho*/ 1.0, /*T*/ 1.0, /*q*/ 1e-15,
                        /*r*/ 0.2, /*thickness*/ 0.0, /*Y42*/ 0.0);

    const int ic = g.nx / 2, jc = g.ny / 2, kc = g.nz / 2;
    EXPECT_NEAR(U[RHO](ic, jc, kc), 1.0, 1e-6);
}
