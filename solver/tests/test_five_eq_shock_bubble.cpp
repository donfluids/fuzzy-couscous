#include <gtest/gtest.h>

#include "bc/BC.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "numerics/RHS.hpp"
#include "numerics/RK3.hpp"
#include "physics/EOS.hpp"
#include "physics/MixtureEOS.hpp"
#include "physics/Multifluid.hpp"
#include "physics/ViscousFlux.hpp"

#include <algorithm>
#include <cmath>

using namespace blast;

// Smoke test for the five-equation model with a SHOCK present: a high-pressure
// driver launches a shock that crosses an embedded material interface (jump in
// volume fraction + density). Exercises shock capturing (WENO via the Ducros
// sensor) AND the contact treatment together. Must stay finite, positive, and
// volume-fraction-bounded, with the mixture mass consistent (rho == Z1+Z2).
TEST(FiveEqShockBubble, StaysBoundedAndFinite) {
    Grid g;
    g.nx = 200; g.ny = 4; g.nz = 4;
    g.lx = 1.0; g.ly = g.lx / g.nx * 4; g.lz = g.ly;
    g.x0 = g.y0 = g.z0 = 0.0;

    GammaLaw gl; gl.gamma = 1.4; gl.R = 1.0;
    IdealGas eos{gl};

    MixtureEOS mix;
    mix.mode = MixMode::FiveEquation;
    mix.phase[0].kind = PhaseEOS::StiffenedGas; mix.phase[0].sg = {1.4, 0.0};
    mix.phase[1].kind = PhaseEOS::StiffenedGas; mix.phase[1].sg = {1.6, 0.0};

    const Real rho1 = 1.0, rho2 = 3.0, eps = mix.a_floor;
    const Real x_if = 0.6, delta = 3.0 * g.dx();

    State U(g.nx, g.ny, g.nz);
    FiveEqAux aux; aux.allocate(g.nx, g.ny, g.nz, U.ng());
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real x = g.xc(i);
                // phase 0 (left of x_if) -> phase 1 (right), smoothed.
                const Real wph = 0.5 * (1.0 + std::tanh((x_if - x) / delta));  // 1 in phase0
                Real a1 = eps + (1.0 - 2.0 * eps) * wph;
                a1 = std::min(std::max(a1, eps), 1.0 - eps);
                const Real Z1 = a1 * rho1, Z2 = (1.0 - a1) * rho2;
                const Real rho = Z1 + Z2;
                const Real p = (x < 0.3) ? 10.0 : 1.0;       // high-pressure driver
                const Real rhoe = mix.five_eq_rhoe_from_p(a1, Z1, Z2, p);
                U[RHO](i,j,k)  = rho;
                U[RHOU](i,j,k) = 0.0;
                U[RHOV](i,j,k) = 0.0;
                U[RHOW](i,j,k) = 0.0;
                U[RHOE](i,j,k) = rhoe;
                aux.Z1(i,j,k) = Z1; aux.Z2(i,j,k) = Z2; aux.a1(i,j,k) = a1;
            }

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Outflow;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;

    ViscousParams vp; vp.mu = 0.0;
    vp.rho_floor = 1e-10; vp.eint_floor = 1e-10;

    RK3 driver(g.nx, g.ny, g.nz, U.ng());
    for (int s = 0; s < 120; ++s) {
        const Real dt = max_dt_hyperbolic(U, g, eos, 0.4, &aux.a1, &mix, &aux);
        ASSERT_GT(dt, 0.0);
        ASSERT_TRUE(std::isfinite(dt));
        driver.step_5eq(U, aux, g, bc, eos, vp, dt, mix);
    }

    Real a1min = 1e300, a1max = -1e300, rho_consistency = 0.0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real rho = U[RHO](i,j,k);
                ASSERT_TRUE(std::isfinite(rho) && rho > 0.0);
                const Real u = U[RHOU](i,j,k)/rho;
                const Real v = U[RHOV](i,j,k)/rho;
                const Real w = U[RHOW](i,j,k)/rho;
                const Real ke = 0.5 * rho * (u*u + v*v + w*w);
                Real p, c;
                mix.p_c_5eq(aux.a1(i,j,k), aux.Z1(i,j,k), aux.Z2(i,j,k),
                            rho, U[RHOE](i,j,k) - ke, p, c);
                ASSERT_TRUE(std::isfinite(p)) << "p not finite at i=" << i;
                EXPECT_GT(p, 0.0);
                a1min = std::min(a1min, aux.a1(i,j,k));
                a1max = std::max(a1max, aux.a1(i,j,k));
                rho_consistency = std::max(rho_consistency,
                    std::fabs(rho - (aux.Z1(i,j,k) + aux.Z2(i,j,k))));
            }
    EXPECT_GE(a1min, -1e-9);
    EXPECT_LE(a1max, 1.0 + 1e-9);
    EXPECT_LT(rho_consistency, 1e-9);
}
