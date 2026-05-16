#include <gtest/gtest.h>

#include "core/Config.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "diagnostics/Statistics.hpp"
#include "ic/Canonical.hpp"
#include "physics/EOS.hpp"

#include <cmath>

using namespace blast;

namespace {

void fill_state(State& U, const Grid& g, const IdealGas& eos, Real rho0, Real p0,
                Real (*ufn)(Real, Real, Real),
                Real (*vfn)(Real, Real, Real),
                Real (*wfn)(Real, Real, Real)) {
    for (int k = -U.ng(); k < g.nz + U.ng(); ++k)
        for (int j = -U.ng(); j < g.ny + U.ng(); ++j)
            for (int i = -U.ng(); i < g.nx + U.ng(); ++i) {
                const Real x = g.xc(i), y = g.yc(j), z = g.zc(k);
                set_from_primitive(U, i, j, k, eos, rho0,
                                   ufn(x, y, z), vfn(x, y, z), wfn(x, y, z), p0);
            }
}

Real u_const(Real, Real, Real) { return 2.0; }
Real v_const(Real, Real, Real) { return 0.0; }
Real w_const(Real, Real, Real) { return 0.0; }

Real u_tg(Real x, Real y, Real z) { return  std::sin(x) * std::cos(y) * std::cos(z); }
Real v_tg(Real x, Real y, Real z) { return -0.5 * std::cos(x) * std::sin(y) * std::cos(z); }
Real w_tg(Real x, Real y, Real z) { return -0.5 * std::cos(x) * std::cos(y) * std::sin(z); }

Real u_compress(Real x, Real, Real) { return std::sin(x); }
Real v_compress(Real, Real y, Real) { return std::sin(y); }
Real w_compress(Real, Real, Real z) { return std::sin(z); }

}  // namespace

TEST(VelocityStats, UniformFlowHasZeroFluctuation) {
    Grid g; g.nx = g.ny = g.nz = 16; g.lx = g.ly = g.lz = 1.0;
    IdealGas eos{GammaLaw{}};
    State U(g.nx, g.ny, g.nz);
    fill_state(U, g, eos, 1.0, 1.0, u_const, v_const, w_const);

    auto s = velocity_stats(U, eos);
    EXPECT_NEAR(s.u_mean[0], 2.0, 1e-12);
    EXPECT_NEAR(s.u_mean[1], 0.0, 1e-12);
    EXPECT_NEAR(s.u_mean[2], 0.0, 1e-12);
    EXPECT_LT(s.u_rms, 1e-12);
    EXPECT_LT(s.tke,   1e-12);
    EXPECT_NEAR(s.ke_total, 2.0, 1e-12);    // (1/2) * 1 * 2^2
}

TEST(Dissipation, TaylorGreenSolenoidalOnly) {
    // Solenoidal field on [0, 2 pi]^3, div u = 0 by construction.
    // omega = curl u; |omega|^2 averaged over the box is computable analytically.
    Grid g; g.nx = g.ny = g.nz = 32;
    g.lx = g.ly = g.lz = 2.0 * M_PI;
    IdealGas eos{GammaLaw{}};
    State U(g.nx, g.ny, g.nz);
    fill_state(U, g, eos, 1.0, 1.0, u_tg, v_tg, w_tg);

    ViscousParams vp; vp.mu = 0.1; vp.prandtl = 0.71;
    auto b = dissipation_budget(U, g, eos, vp);

    EXPECT_LT(b.div2_mean, 1e-12);
    EXPECT_GT(b.omega2_mean, 0.5);
    EXPECT_GT(b.eps_sol, 0.0);
    EXPECT_LT(b.eps_dil, 1e-12);
    EXPECT_NEAR(b.eps_total, b.eps_sol, 1e-10);
}

TEST(Dissipation, IrrotationalDilatationalOnly) {
    // u_i = sin(x_i) -> div nonzero, vorticity zero.
    Grid g; g.nx = g.ny = g.nz = 32;
    g.lx = g.ly = g.lz = 2.0 * M_PI;
    IdealGas eos{GammaLaw{}};
    State U(g.nx, g.ny, g.nz);
    fill_state(U, g, eos, 1.0, 1.0, u_compress, v_compress, w_compress);

    ViscousParams vp; vp.mu = 0.1; vp.prandtl = 0.71;
    auto b = dissipation_budget(U, g, eos, vp);

    EXPECT_LT(b.omega2_mean, 1e-12);
    EXPECT_GT(b.div2_mean, 0.5);
    EXPECT_LT(b.eps_sol, 1e-12);
    EXPECT_GT(b.eps_dil, 0.0);
    // div(u) = cos(x)+cos(y)+cos(z); <div^2> = 3 * <cos^2> = 3/2.
    // On 32^3 the 6th-order operator gives truncation error ~ 1e-6.
    EXPECT_NEAR(b.div2_mean, 1.5, 1e-4);
}

TEST(VelocityStats, MachNumberMatchesAnalytic) {
    Grid g; g.nx = g.ny = g.nz = 32;
    g.lx = g.ly = g.lz = 2.0 * M_PI;
    IdealGas eos{GammaLaw{}};
    State U(g.nx, g.ny, g.nz);
    fill_state(U, g, eos, 1.0, 1.0, u_tg, v_tg, w_tg);

    auto s = velocity_stats(U, eos);
    // For (rho=1, p=1, gamma=1.4): c = sqrt(1.4)
    const Real c_expected = std::sqrt(1.4);
    EXPECT_NEAR(s.c_mean, c_expected, 1e-10);
    // Mean velocity should be zero by symmetry of TG sinusoids.
    for (int d = 0; d < 3; ++d) EXPECT_NEAR(s.u_mean[d], 0.0, 1e-12);
    EXPECT_GT(s.M_t, 0.0);
    EXPECT_LT(s.M_t, 1.0);  // subsonic
}
