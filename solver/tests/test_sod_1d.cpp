#include <gtest/gtest.h>

#include "bc/BC.hpp"
#include "core/Config.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "ic/Canonical.hpp"
#include "numerics/RHS.hpp"
#include "numerics/RK3.hpp"
#include "physics/EOS.hpp"
#include "physics/RiemannExact.hpp"

#include <algorithm>
#include <cmath>

using namespace blast;

namespace {

// Run Sod and return L1 errors on rho, u, p, plus exact reference vectors.
struct Result {
    Real l1_rho, l1_u, l1_p;
    Real min_rho, min_p;
};

Result run_sod(int N, Real t_end) {
    Grid g;
    g.nx = N; g.ny = 4; g.nz = 4;
    g.lx = 1.0; g.ly = g.lx / N * 4; g.lz = g.ly;
    g.x0 = 0.0; g.y0 = 0.0; g.z0 = 0.0;

    IdealGas eos{GammaLaw{}};
    State U(g.nx, g.ny, g.nz);
    ic_sod_x(U, g, eos);

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Outflow;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;

    RK3 driver(g.nx, g.ny, g.nz, U.ng());

    Real t = 0.0;
    while (t < t_end) {
        Real dt = max_dt_hyperbolic(U, g, eos, 0.4);
        if (t + dt > t_end) dt = t_end - t;
        driver.step(U, g, bc, eos, dt);
        t += dt;
    }

    // Compare against exact Riemann solution.
    riemann::LR L{1.0, 0.0, 1.0};
    riemann::LR R{0.125, 0.0, 0.1};
    riemann::StarState star = riemann::star_state(L, R, eos.eos);

    const int jmid = g.ny / 2;
    const int kmid = g.nz / 2;
    const Real x_mid = g.x0 + 0.5 * g.lx;

    Real e_rho = 0, e_u = 0, e_p = 0;
    Real min_rho = 1e30, min_p = 1e30;
    for (int i = 0; i < g.nx; ++i) {
        const Real x = g.xc(i);
        const Real s = (x - x_mid) / t_end;
        riemann::LR ex = riemann::sample(L, R, star, s, eos.eos);

        const Real rho = U[RHO](i, jmid, kmid);
        const Real u   = U[RHOU](i, jmid, kmid) / rho;
        const Real ke  = 0.5 * rho * u * u;
        const Real e_int = U[RHOE](i, jmid, kmid) - ke;
        const Real p   = eos.pressure(rho, e_int);
        e_rho += std::fabs(rho - ex.rho);
        e_u   += std::fabs(u   - ex.u);
        e_p   += std::fabs(p   - ex.p);
        min_rho = std::min(min_rho, rho);
        min_p   = std::min(min_p,   p);
    }
    return {e_rho / g.nx, e_u / g.nx, e_p / g.nx, min_rho, min_p};
}

}  // namespace

TEST(Sod1D, AccuracyAndPositivity) {
    auto r = run_sod(200, 0.2);
    // Reference: 5th-order WENO on 200 cells typically gives L1(rho) ~ 1.5e-3
    // and L1(u), L1(p) ~ a few e-3. Hybrid with Ducros sensor should be in the
    // same ballpark. Set lenient envelopes; tighten after baseline established.
    EXPECT_LT(r.l1_rho, 0.02);
    EXPECT_LT(r.l1_u,   0.04);
    EXPECT_LT(r.l1_p,   0.02);
    EXPECT_GT(r.min_rho, 0.0);
    EXPECT_GT(r.min_p,   0.0);
}

TEST(Sod1D, ConvergesWithRefinement) {
    auto r1 = run_sod(100, 0.2);
    auto r2 = run_sod(200, 0.2);
    // Discontinuities limit asymptotic order to ~1 in L1; expect modest gain.
    EXPECT_LT(r2.l1_rho, r1.l1_rho);
    EXPECT_LT(r2.l1_u,   r1.l1_u);
    EXPECT_LT(r2.l1_p,   r1.l1_p);
}
