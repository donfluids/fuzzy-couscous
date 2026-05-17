#include <gtest/gtest.h>

#include "bc/BC.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "physics/EOS.hpp"
#include "physics/Forcing.hpp"

#include <cmath>

using namespace blast;

namespace {

State quiescent(int N, Grid& g_out, Real rho = 1.0, Real p = 1.0) {
    g_out.nx = g_out.ny = g_out.nz = N;
    g_out.lx = g_out.ly = g_out.lz = 2.0 * M_PI;
    g_out.x0 = g_out.y0 = g_out.z0 = 0.0;

    State U(N, N, N);
    auto& rho_a = U[RHO];
    auto& mx = U[RHOU]; auto& my = U[RHOV]; auto& mz = U[RHOW];
    auto& en = U[RHOE];
    const Real gam = 1.4;
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                rho_a(i, j, k) = rho;
                mx(i, j, k) = 0;
                my(i, j, k) = 0;
                mz(i, j, k) = 0;
                en(i, j, k) = p / (gam - 1.0);
            }
    BCSet bc;
    bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;
    apply_bcs(U, bc);
    return U;
}

Real cell_divergence(const State& U, const Grid& g, int i, int j, int k) {
    const int N = g.nx;
    auto wrap = [&](int x) { return (x % N + N) % N; };
    auto& rho = U[RHO];
    auto& mx = U[RHOU];
    auto& my = U[RHOV];
    auto& mz = U[RHOW];

    auto u_at = [&](int ii, int jj, int kk) {
        const int wi = wrap(ii), wj = wrap(jj), wk = wrap(kk);
        return mx(wi, wj, wk) / rho(wi, wj, wk);
    };
    auto v_at = [&](int ii, int jj, int kk) {
        const int wi = wrap(ii), wj = wrap(jj), wk = wrap(kk);
        return my(wi, wj, wk) / rho(wi, wj, wk);
    };
    auto w_at = [&](int ii, int jj, int kk) {
        const int wi = wrap(ii), wj = wrap(jj), wk = wrap(kk);
        return mz(wi, wj, wk) / rho(wi, wj, wk);
    };

    const Real du_dx = (u_at(i + 1, j, k) - u_at(i - 1, j, k)) / (2.0 * g.dx());
    const Real dv_dy = (v_at(i, j + 1, k) - v_at(i, j - 1, k)) / (2.0 * g.dy());
    const Real dw_dz = (w_at(i, j, k + 1) - w_at(i, j, k - 1)) / (2.0 * g.dz());
    return du_dx + dv_dy + dw_dz;
}

}  // namespace

// 1. The applied force has zero mean (no k=0 mode). After one apply on a
//    zero-velocity field, integrated momentum components stay near zero.
TEST(SpectralForcing, ZeroMeanMomentum) {
    Grid g;
    State U = quiescent(32, g);

    SpectralForcing::Params p;
    p.k_lo = 1; p.k_hi = 3; p.eps_target = 0.05;
    p.T_corr = 1.0; p.seed = 42;
    SpectralForcing fc(g, p);
    fc.evolve_ou(0.01);
    fc.apply(U, g, 0.01);

    Real px = 0, py = 0, pz = 0;
    auto& mx = U[RHOU]; auto& my = U[RHOV]; auto& mz = U[RHOW];
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                px += mx(i, j, k);
                py += my(i, j, k);
                pz += mz(i, j, k);
            }
    const Real N3 = static_cast<Real>(g.nx) * g.ny * g.nz;
    EXPECT_NEAR(px / N3, 0.0, 1e-12);
    EXPECT_NEAR(py / N3, 0.0, 1e-12);
    EXPECT_NEAR(pz / N3, 0.0, 1e-12);
}

// 2. The momentum injection field is divergence-free by construction
//    (helical basis projects out the k-parallel component). After one
//    apply on a zero-velocity field, the velocity is dt * f -- and div f
//    must be zero in the continuum. Discrete central-difference div(u)
//    measured on the periodic mesh is suppressed by the truncation error,
//    not the helical projection, so we check that the dilatation is small
//    relative to the velocity scale.
TEST(SpectralForcing, ApproximatelyDivergenceFreeInjection) {
    Grid g;
    State U = quiescent(48, g);

    SpectralForcing::Params p;
    p.k_lo = 1; p.k_hi = 3; p.eps_target = 0.1;
    p.T_corr = 1.0; p.seed = 7;
    SpectralForcing fc(g, p);
    fc.evolve_ou(0.01);
    fc.apply(U, g, 0.01);

    // ||u||_rms
    Real q2 = 0;
    auto& rho = U[RHO];
    auto& mx = U[RHOU]; auto& my = U[RHOV]; auto& mz = U[RHOW];
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real r = rho(i, j, k);
                const Real u = mx(i, j, k) / r;
                const Real v = my(i, j, k) / r;
                const Real w = mz(i, j, k) / r;
                q2 += u * u + v * v + w * w;
            }
    const Real N3 = static_cast<Real>(g.nx) * g.ny * g.nz;
    const Real u_rms = std::sqrt(q2 / (3 * N3));
    ASSERT_GT(u_rms, 1e-10) << "forcing produced no velocity";

    // |div u|_rms / (u_rms / dx) should be << 1 for the smooth low-k field.
    Real d2 = 0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real div = cell_divergence(U, g, i, j, k);
                d2 += div * div;
            }
    const Real div_rms = std::sqrt(d2 / N3);
    // For low-k modes, second-order central FD recovers div to O(dx^2 k^2).
    // With k_hi=3 and dx ~ 0.13, expect div_rms ~ (k dx)^2 u_rms ~ 0.16 u_rms.
    EXPECT_LT(div_rms, 0.2 * (u_rms / g.dx()))
        << "div_rms=" << div_rms << ", u_rms=" << u_rms << ", dx=" << g.dx();
}

// 3. Constant-power injection: after settling, the measured global injection
//    power matches eps_target to high relative precision. Run a few warm-up
//    steps so the velocity becomes correlated with the force.
TEST(SpectralForcing, InjectionPowerLocksToTarget) {
    Grid g;
    State U = quiescent(32, g);

    SpectralForcing::Params p;
    p.k_lo = 1; p.k_hi = 3; p.eps_target = 0.5;
    p.T_corr = 1.0; p.seed = 99;
    SpectralForcing fc(g, p);

    const Real dt = 0.01;
    for (int s = 0; s < 50; ++s) {
        fc.evolve_ou(dt);
        fc.apply(U, g, dt);
    }
    EXPECT_NEAR(fc.last_inject_power(), p.eps_target, 1e-10)
        << "measured=" << fc.last_inject_power();
}

// 4. Energy balance: in absence of any other RHS, after one apply step the
//    increase in kinetic energy density (per unit mass) equals
//    dt * eps_target to leading order.
TEST(SpectralForcing, KineticEnergyGrowthMatchesEpsDt) {
    Grid g;
    State U = quiescent(32, g, /*rho=*/1.0, /*p=*/1.0);

    SpectralForcing::Params p;
    p.k_lo = 1; p.k_hi = 3; p.eps_target = 0.1;
    p.T_corr = 1.0; p.seed = 17;
    SpectralForcing fc(g, p);

    auto ke_per_mass = [&]() {
        Real q2 = 0, mass = 0;
        auto& rho = U[RHO];
        auto& mx = U[RHOU]; auto& my = U[RHOV]; auto& mz = U[RHOW];
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j)
                for (int i = 0; i < g.nx; ++i) {
                    const Real r = rho(i, j, k);
                    const Real u = mx(i, j, k) / r;
                    const Real v = my(i, j, k) / r;
                    const Real w = mz(i, j, k) / r;
                    q2 += 0.5 * r * (u*u + v*v + w*w);
                    mass += r;
                }
        return q2 / mass;
    };

    // Warm up so |u| isn't tiny (otherwise the scale clamp kicks in).
    const Real dt = 0.005;
    for (int s = 0; s < 20; ++s) { fc.evolve_ou(dt); fc.apply(U, g, dt); }

    const Real ke0 = ke_per_mass();
    fc.evolve_ou(dt); fc.apply(U, g, dt);
    const Real ke1 = ke_per_mass();
    const Real dKE = ke1 - ke0;

    // Predicted: dKE ~ eps_target * dt (Lie splitting). Allow 2x tolerance
    // because the OU draw at this step is non-deterministic with respect
    // to <u . f> beyond the rescaling.
    EXPECT_NEAR(dKE, p.eps_target * dt, 0.5 * p.eps_target * dt)
        << "dKE=" << dKE << ", expected ~" << p.eps_target * dt;
}

// 5. Determinism: two forcing objects with the same seed produce identical
//    OU evolutions and identical force fields at every step.
TEST(SpectralForcing, SeedDeterminism) {
    Grid g;
    State U1 = quiescent(16, g);
    State U2 = quiescent(16, g);

    SpectralForcing::Params p;
    p.k_lo = 1; p.k_hi = 3; p.eps_target = 0.1;
    p.T_corr = 1.0; p.seed = 314159;
    SpectralForcing f1(g, p);
    SpectralForcing f2(g, p);

    const Real dt = 0.01;
    for (int s = 0; s < 10; ++s) {
        f1.evolve_ou(dt); f1.apply(U1, g, dt);
        f2.evolve_ou(dt); f2.apply(U2, g, dt);
    }

    Real maxd = 0;
    for (int v = 0; v < NCONS; ++v)
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j)
                for (int i = 0; i < g.nx; ++i)
                    maxd = std::max(maxd, std::fabs(U1[v](i, j, k) - U2[v](i, j, k)));
    EXPECT_LT(maxd, 1e-14);
}
