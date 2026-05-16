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

namespace {

constexpr Real M_PI_ = 3.14159265358979323846;

Grid tgv_grid(int N) {
    Grid g;
    g.nx = g.ny = g.nz = N;
    g.lx = g.ly = g.lz = 2.0 * M_PI_;
    g.x0 = g.y0 = g.z0 = 0.0;
    return g;
}

}  // namespace

TEST(TGV, InitialEnstrophyMatchesAnalytic) {
    // <|omega|^2>_{TG, t=0} = (3/4) V0^2 (independent of M_0).
    auto g = tgv_grid(32);
    IdealGas eos{GammaLaw{}};
    State U(g.nx, g.ny, g.nz);

    const Real V0 = 1.0, rho_0 = 1.0, M_0 = 0.1;
    ic_taylor_green_3d(U, g, eos, V0, rho_0, M_0);

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;
    apply_bcs(U, bc);

    ViscousParams vp; vp.mu = 1.0 / 1600.0;
    auto b = dissipation_budget(U, g, eos, vp);

    EXPECT_NEAR(b.omega2_mean, 0.75 * V0 * V0, 1e-3);
    EXPECT_LT(b.div2_mean, 1e-3);
    EXPECT_NEAR(b.eps_sol, vp.mu / rho_0 * 0.75 * V0 * V0, 1e-6);
}

TEST(TGV, RunsAFewStepsAndDissipationGrows) {
    auto g = tgv_grid(32);
    IdealGas eos{GammaLaw{}};
    State U(g.nx, g.ny, g.nz);
    const Real V0 = 1.0, rho_0 = 1.0, M_0 = 0.1;
    ic_taylor_green_3d(U, g, eos, V0, rho_0, M_0);

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;
    apply_bcs(U, bc);

    RK3 driver(g.nx, g.ny, g.nz, U.ng());

    ViscousParams vp; vp.mu = 1.0 / 1600.0;
    auto b0 = dissipation_budget(U, g, eos, vp);
    const Real ke0 = velocity_stats(U, eos).ke_total;
    EXPECT_NEAR(ke0, 0.125 * rho_0 * V0 * V0, 1e-3);  // analytic

    // Step until t ~ 1.0 (well before the peak dissipation at t ~ 9, but
    // long enough that vortex stretching has begun).
    const Real t_end = 1.0;
    Real t = 0.0;
    while (t < t_end) {
        Real dt_hyp = max_dt_hyperbolic(U, g, eos, 0.4);
        Real dt_vis = max_dt_viscous(U, g, vp, 0.25);
        Real dt = std::min(dt_hyp, dt_vis);
        if (t + dt > t_end) dt = t_end - t;
        driver.step(U, g, bc, eos, vp, dt);
        t += dt;
    }

    auto b1 = dissipation_budget(U, g, eos, vp);
    EXPECT_GT(b1.omega2_mean, b0.omega2_mean) << "vortex stretching should grow enstrophy";

    auto stats = velocity_stats(U, eos);
    EXPECT_GT(stats.ke_total, 0.0);
    EXPECT_LT(stats.ke_total, ke0);     // some decay
}
