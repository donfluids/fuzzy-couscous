#include <gtest/gtest.h>

#include "bc/BC.hpp"
#include "core/Config.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "ic/Canonical.hpp"
#include "numerics/RHS.hpp"
#include "numerics/RK3.hpp"
#include "physics/EOS.hpp"

#include <cmath>

using namespace blast;

namespace {

// Fill the full padded region with a single-mode density wave at quiescent
// flow, uniform pressure p_0 -- so only the density variable carries a
// nonzero Laplacian; momentum / energy are constants and should not be
// touched by the operator.
void fill_density_mode(State& U, const Grid& g, const IdealGas& eos,
                       Real rho_0, Real A, Real k0, Real p_0) {
    for (int k = -U.ng(); k < g.nz + U.ng(); ++k)
        for (int j = -U.ng(); j < g.ny + U.ng(); ++j)
            for (int i = -U.ng(); i < g.nx + U.ng(); ++i) {
                const Real x = g.xc(i);
                const Real rho = rho_0 + A * std::sin(k0 * x);
                set_from_primitive(U, i, j, k, eos, rho, 0.0, 0.0, 0.0, p_0);
            }
}

}  // namespace

// nabla^4 sin(k0 x) = k0^4 sin(k0 x), so the hyperdissipation operator
// -nu_h nabla^4 should produce RHS[RHO] = -nu_h k0^4 (rho - rho_0) at every
// cell. Compare cell-wise.
TEST(Hyperdissipation, SingleModeOperatorMatchesNuhK4) {
    const int N = 32;
    Grid g; g.nx = g.ny = g.nz = N;
    g.lx = g.ly = g.lz = 2.0 * M_PI;
    g.x0 = g.y0 = g.z0 = 0.0;

    IdealGas eos{GammaLaw{}};
    State U(N, N, N);
    const Real rho_0 = 1.0, A = 0.1, k0 = 1.0, p_0 = 1.0;
    fill_density_mode(U, g, eos, rho_0, A, k0, p_0);

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;
    apply_bcs(U, bc);

    State Rhs(N, N, N);
    for (int v = 0; v < NCONS; ++v) Rhs.fill(v, 0.0);

    ViscousParams vp;
    vp.mu = 0.0;                   // disable viscous
    vp.hyper_coeff = 0.01;
    add_rhs_viscous(U, g, eos, vp, Rhs);

    const Real k4 = std::pow(k0, 4);
    Real l2_err = 0.0, l2_ref = 0.0;
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const Real x = g.xc(i);
                const Real expected = -vp.hyper_coeff * k4 * A * std::sin(k0 * x);
                const Real got = Rhs[RHO](i, j, k);
                l2_err += (got - expected) * (got - expected);
                l2_ref += expected * expected;
            }
    const Real rel = std::sqrt(l2_err / l2_ref);
    // 6th-order stencil composed twice on a k0*dx = 0.196 mode gives ~0.3%
    // truncation. Allow 1%.
    EXPECT_LT(rel, 1e-2) << "relative L2 = " << rel;

    // No other conserved variable has a nontrivial Laplacian here, so their
    // RHS contributions from hyperdissipation must vanish to roundoff.
    Real max_mom = 0.0, max_E = 0.0;
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                max_mom = std::max({max_mom,
                                    std::fabs(Rhs[RHOU](i, j, k)),
                                    std::fabs(Rhs[RHOV](i, j, k)),
                                    std::fabs(Rhs[RHOW](i, j, k))});
                max_E   = std::max(max_E, std::fabs(Rhs[RHOE](i, j, k)));
            }
    EXPECT_LT(max_mom, 1e-12);
    EXPECT_LT(max_E,   1e-12);
}

// Integrating the single-mode density wave forward in time should decay its
// amplitude as exp(-lambda t) where lambda is the discrete operator
// eigenvalue at this mode. We extract lambda from a one-shot RHS evaluation,
// not from the analytic nu_h k^4 -- that isolates time-integration error
// from operator-truncation error.
TEST(Hyperdissipation, TimeIntegrationMatchesDiscreteEigenvalueDecay) {
    const int N = 16;
    Grid g; g.nx = g.ny = g.nz = N;
    g.lx = g.ly = g.lz = 2.0 * M_PI;
    g.x0 = g.y0 = g.z0 = 0.0;

    IdealGas eos{GammaLaw{}};
    State U(N, N, N);
    const Real rho_0 = 1.0, A = 0.1, k0 = 1.0, p_0 = 1.0;
    fill_density_mode(U, g, eos, rho_0, A, k0, p_0);

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;
    apply_bcs(U, bc);

    ViscousParams vp;
    vp.mu = 0.0;
    // RK3 explicit stability requires dt * nu_h * k_max^4 < ~2.5. For N=16,
    // k_max = 8, k_max^4 = 4096. Use nu_h = 0.5 with dt = 0.001 -> stability
    // margin dt * nu_h * k_max^4 = 2.05 (safe). Damping rate at k0=1 is
    // nu_h * k0^4 = 0.5, so over t_end=2.0 amplitude shrinks by exp(-1) ~ 0.37.
    vp.hyper_coeff = 0.5;

    // First: extract the discrete eigenvalue lambda at mode k0 from a single
    // RHS evaluation. Rhs[RHO](i,0,0) = -lambda * (rho(x_i) - rho_0).
    State Rhs(N, N, N);
    for (int v = 0; v < NCONS; ++v) Rhs.fill(v, 0.0);
    add_rhs_viscous(U, g, eos, vp, Rhs);

    Real num = 0.0, denom = 0.0;
    for (int i = 0; i < N; ++i) {
        const Real x = g.xc(i);
        const Real s = std::sin(k0 * x);
        num   += -Rhs[RHO](i, 0, 0) * s;     // -Rhs ~ lambda * A sin
        denom += A * s * s;
    }
    const Real lambda_discrete = num / denom;

    // Time-integrate under hyperdissipation alone.
    RK3 driver(N, N, N, U.ng());
    const Real t_end = 2.0;
    const Real dt = 1.0e-3;          // RK3 stability: dt*nu_h*k_max^4 = 2.05
    Real t = 0.0;
    while (t < t_end) {
        Real step_dt = std::min(dt, t_end - t);
        driver.step(U, g, bc, eos, vp, step_dt);
        t += step_dt;
    }

    // Project final state onto the sin(k0 x) mode along x = i, j=k=0 slab.
    Real num2 = 0.0, denom2 = 0.0;
    for (int i = 0; i < N; ++i) {
        const Real x = g.xc(i);
        const Real s = std::sin(k0 * x);
        num2 += (U[RHO](i, 0, 0) - rho_0) * s;
        denom2 += s * s;
    }
    const Real A_meas = num2 / denom2;
    const Real A_pred = A * std::exp(-lambda_discrete * t_end);

    const Real rel = std::fabs(A_meas - A_pred) / std::fabs(A_pred);
    EXPECT_LT(rel, 1e-3)
        << "A_meas=" << A_meas << " A_pred=" << A_pred
        << " lambda_disc=" << lambda_discrete;
}

// (nabla^2)^3 sin(k0 x) = -k0^6 sin(k0 x), and the higher-power hyperdissipation
// is +nu_h6 (nabla^2)^3 U, so RHS[RHO] = -nu_h6 k0^6 (rho - rho_0).
TEST(Hyperdissipation, Nabla6SingleModeOperatorMatchesNuh6K6) {
    const int N = 32;
    Grid g; g.nx = g.ny = g.nz = N;
    g.lx = g.ly = g.lz = 2.0 * M_PI;
    g.x0 = g.y0 = g.z0 = 0.0;

    IdealGas eos{GammaLaw{}};
    State U(N, N, N);
    const Real rho_0 = 1.0, A = 0.1, k0 = 1.0, p_0 = 1.0;
    fill_density_mode(U, g, eos, rho_0, A, k0, p_0);

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;
    apply_bcs(U, bc);

    State Rhs(N, N, N);
    for (int v = 0; v < NCONS; ++v) Rhs.fill(v, 0.0);

    ViscousParams vp;
    vp.mu = 0.0;
    vp.hyper_coeff = 0.0;          // disable nabla^4 path
    vp.hyper6_coeff = 0.01;
    add_rhs_viscous(U, g, eos, vp, Rhs);

    const Real k6 = std::pow(k0, 6);
    Real l2_err = 0.0, l2_ref = 0.0;
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const Real x = g.xc(i);
                const Real expected = -vp.hyper6_coeff * k6 * A * std::sin(k0 * x);
                const Real got = Rhs[RHO](i, j, k);
                l2_err += (got - expected) * (got - expected);
                l2_ref += expected * expected;
            }
    const Real rel = std::sqrt(l2_err / l2_ref);
    // 4th-order Laplacian composed three times on a k0*dx = 0.196 mode is
    // dominated by the per-stencil O((k h)^4 / 90) truncation; three
    // compositions stay well under 0.1%. Allow 1% for headroom.
    EXPECT_LT(rel, 1e-2) << "relative L2 = " << rel;

    Real max_mom = 0.0, max_E = 0.0;
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                max_mom = std::max({max_mom,
                                    std::fabs(Rhs[RHOU](i, j, k)),
                                    std::fabs(Rhs[RHOV](i, j, k)),
                                    std::fabs(Rhs[RHOW](i, j, k))});
                max_E   = std::max(max_E, std::fabs(Rhs[RHOE](i, j, k)));
            }
    EXPECT_LT(max_mom, 1e-12);
    EXPECT_LT(max_E,   1e-12);
}

// Both nabla^4 and nabla^6 paths active simultaneously: the operator is
// linear in U, so the RHS must equal the sum of the two single-power
// contributions.
TEST(Hyperdissipation, Nabla4AndNabla6Superpose) {
    const int N = 32;
    Grid g; g.nx = g.ny = g.nz = N;
    g.lx = g.ly = g.lz = 2.0 * M_PI;
    g.x0 = g.y0 = g.z0 = 0.0;

    IdealGas eos{GammaLaw{}};
    State U(N, N, N);
    const Real rho_0 = 1.0, A = 0.1, k0 = 1.0, p_0 = 1.0;
    fill_density_mode(U, g, eos, rho_0, A, k0, p_0);

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;
    apply_bcs(U, bc);

    auto eval = [&](Real c4, Real c6, State& out) {
        for (int v = 0; v < NCONS; ++v) out.fill(v, 0.0);
        ViscousParams vp;
        vp.mu = 0.0;
        vp.hyper_coeff  = c4;
        vp.hyper6_coeff = c6;
        add_rhs_viscous(U, g, eos, vp, out);
    };

    State R4(N, N, N), R6(N, N, N), Rboth(N, N, N);
    eval(0.01, 0.0,  R4);
    eval(0.0,  0.01, R6);
    eval(0.01, 0.01, Rboth);

    Real max_diff = 0.0;
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const Real sum = R4[RHO](i, j, k) + R6[RHO](i, j, k);
                max_diff = std::max(max_diff,
                                    std::fabs(Rboth[RHO](i, j, k) - sum));
            }
    EXPECT_LT(max_diff, 1e-12);
}

// Spectral path: the operator is exact in spectral space, so on a pure
// single-mode sin(k0 x) the result should match -nu_h k0^4 sin(k0 x) at
// roundoff. (FD path got ~0.3% truncation; spectral should be ~1e-13.)
TEST(Hyperdissipation, SpectralNabla4MatchesAnalyticOnSingleMode) {
    const int N = 32;
    Grid g; g.nx = g.ny = g.nz = N;
    g.lx = g.ly = g.lz = 2.0 * M_PI;
    g.x0 = g.y0 = g.z0 = 0.0;

    IdealGas eos{GammaLaw{}};
    State U(N, N, N);
    const Real rho_0 = 1.0, A = 0.1, k0 = 1.0, p_0 = 1.0;
    fill_density_mode(U, g, eos, rho_0, A, k0, p_0);

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;
    apply_bcs(U, bc);

    State Rhs(N, N, N);
    for (int v = 0; v < NCONS; ++v) Rhs.fill(v, 0.0);

    ViscousParams vp;
    vp.mu = 0.0;
    vp.hyper_coeff = 0.01;
    vp.hyper6_coeff = 0.0;
    vp.hyper_method = HyperMethod::Pseudospectral;
    add_rhs_viscous(U, g, eos, vp, Rhs);

    const Real k4 = std::pow(k0, 4);
    Real l2_err = 0.0, l2_ref = 0.0;
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const Real x = g.xc(i);
                const Real expected = -vp.hyper_coeff * k4 * A * std::sin(k0 * x);
                const Real got = Rhs[RHO](i, j, k);
                l2_err += (got - expected) * (got - expected);
                l2_ref += expected * expected;
            }
    // FFT roundtrip roundoff amplified by |k|^4 at the Nyquist; far below
    // the FD test's 1e-2, still demonstrates the operator is exact-on-modes.
    EXPECT_LT(std::sqrt(l2_err / l2_ref), 1e-10);
}

TEST(Hyperdissipation, SpectralNabla6MatchesAnalyticOnSingleMode) {
    const int N = 32;
    Grid g; g.nx = g.ny = g.nz = N;
    g.lx = g.ly = g.lz = 2.0 * M_PI;
    g.x0 = g.y0 = g.z0 = 0.0;

    IdealGas eos{GammaLaw{}};
    State U(N, N, N);
    const Real rho_0 = 1.0, A = 0.1, k0 = 1.0, p_0 = 1.0;
    fill_density_mode(U, g, eos, rho_0, A, k0, p_0);

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;
    apply_bcs(U, bc);

    State Rhs(N, N, N);
    for (int v = 0; v < NCONS; ++v) Rhs.fill(v, 0.0);

    ViscousParams vp;
    vp.mu = 0.0;
    vp.hyper_coeff = 0.0;
    vp.hyper6_coeff = 0.01;
    vp.hyper_method = HyperMethod::Pseudospectral;
    add_rhs_viscous(U, g, eos, vp, Rhs);

    const Real k6 = std::pow(k0, 6);
    Real l2_err = 0.0, l2_ref = 0.0;
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const Real x = g.xc(i);
                const Real expected = -vp.hyper6_coeff * k6 * A * std::sin(k0 * x);
                const Real got = Rhs[RHO](i, j, k);
                l2_err += (got - expected) * (got - expected);
                l2_ref += expected * expected;
            }
    // |k|^6 amplifies Nyquist roundoff ~250x more than |k|^4, so tolerance
    // is correspondingly looser; still ~7 orders tighter than the FD test.
    EXPECT_LT(std::sqrt(l2_err / l2_ref), 1e-7);
}

// Spectral and FD must agree on a smooth field to within the FD operator's
// truncation error. At N=32 with k0=1 the 6th-order composed Laplacian
// gives ~0.3% on the existing single-mode test; we allow 1% here.
TEST(Hyperdissipation, SpectralAndFdAgreeOnSmoothField) {
    const int N = 32;
    Grid g; g.nx = g.ny = g.nz = N;
    g.lx = g.ly = g.lz = 2.0 * M_PI;
    g.x0 = g.y0 = g.z0 = 0.0;

    IdealGas eos{GammaLaw{}};
    State U(N, N, N);

    // Density carries a multi-axis mode; momentum/energy are constants.
    const Real rho_0 = 1.0, A = 0.1, k0 = 1.0, p_0 = 1.0;
    for (int k = -U.ng(); k < g.nz + U.ng(); ++k)
        for (int j = -U.ng(); j < g.ny + U.ng(); ++j)
            for (int i = -U.ng(); i < g.nx + U.ng(); ++i) {
                const Real x = g.xc(i);
                const Real y = g.yc(j);
                const Real z = g.zc(k);
                const Real rho = rho_0 + A * (std::sin(k0 * x)
                                            + std::sin(k0 * y)
                                            + std::sin(k0 * z));
                set_from_primitive(U, i, j, k, eos, rho, 0.0, 0.0, 0.0, p_0);
            }

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;
    apply_bcs(U, bc);

    auto eval = [&](HyperMethod m, State& out) {
        for (int v = 0; v < NCONS; ++v) out.fill(v, 0.0);
        ViscousParams vp;
        vp.mu = 0.0;
        vp.hyper_coeff = 0.01;
        vp.hyper6_coeff = 0.0;
        vp.hyper_method = m;
        add_rhs_viscous(U, g, eos, vp, out);
    };

    State R_fd(N, N, N), R_sp(N, N, N);
    eval(HyperMethod::FiniteDifference, R_fd);
    eval(HyperMethod::Pseudospectral,   R_sp);

    Real l2_diff = 0.0, l2_ref = 0.0;
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const Real a = R_fd[RHO](i, j, k);
                const Real b = R_sp[RHO](i, j, k);
                l2_diff += (a - b) * (a - b);
                l2_ref  += b * b;
            }
    EXPECT_LT(std::sqrt(l2_diff / l2_ref), 1e-2);
}

TEST(Hyperdissipation, SpectralZeroCoefficientHasNoEffect) {
    const int N = 16;
    Grid g; g.nx = g.ny = g.nz = N;
    g.lx = g.ly = g.lz = 2.0 * M_PI;
    g.x0 = g.y0 = g.z0 = 0.0;
    IdealGas eos{GammaLaw{}};
    State U(N, N, N);
    fill_density_mode(U, g, eos, 1.0, 0.1, 1.0, 1.0);

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;
    apply_bcs(U, bc);

    State Rhs(N, N, N);
    for (int v = 0; v < NCONS; ++v) Rhs.fill(v, 0.0);

    ViscousParams vp;
    vp.mu = 0.0;
    vp.hyper_coeff = 0.0;
    vp.hyper6_coeff = 0.0;
    vp.hyper_method = HyperMethod::Pseudospectral;
    add_rhs_viscous(U, g, eos, vp, Rhs);

    Real max_abs = 0.0;
    for (int v = 0; v < NCONS; ++v)
        for (int k = 0; k < N; ++k)
            for (int j = 0; j < N; ++j)
                for (int i = 0; i < N; ++i)
                    max_abs = std::max(max_abs, std::fabs(Rhs[v](i, j, k)));
    EXPECT_LT(max_abs, 1e-30);
}

// All-slip-wall pseudospectral path: density follows DCT-II on every
// axis. Lowest cosine mode has k = pi/L. Picking L = pi makes k = 1.
TEST(Hyperdissipation, SpectralSlipWallNabla4OnDensityMode) {
    const int N = 32;
    Grid g; g.nx = g.ny = g.nz = N;
    g.lx = g.ly = g.lz = M_PI;     // makes the lowest cosine mode have k=1
    g.x0 = g.y0 = g.z0 = 0.0;

    IdealGas eos{GammaLaw{}};
    State U(N, N, N);
    const Real rho_0 = 1.0, A = 0.1, p_0 = 1.0;
    // Fill cos(pi * x / L) along x (lowest DCT-II mode); flat in y, z.
    for (int k = -U.ng(); k < g.nz + U.ng(); ++k)
        for (int j = -U.ng(); j < g.ny + U.ng(); ++j)
            for (int i = -U.ng(); i < g.nx + U.ng(); ++i) {
                const Real x = g.xc(i);
                const Real rho = rho_0 + A * std::cos(M_PI * x / g.lx);
                set_from_primitive(U, i, j, k, eos, rho, 0.0, 0.0, 0.0, p_0);
            }

    BCSet bc;
    bc.xlo = bc.xhi = BCType::SlipWall;
    bc.ylo = bc.yhi = BCType::SlipWall;
    bc.zlo = bc.zhi = BCType::SlipWall;
    apply_bcs(U, bc);

    State Rhs(N, N, N);
    for (int v = 0; v < NCONS; ++v) Rhs.fill(v, 0.0);

    ViscousParams vp;
    vp.mu = 0.0;
    vp.hyper_coeff = 0.01;
    vp.hyper6_coeff = 0.0;
    vp.hyper_method = HyperMethod::Pseudospectral;
    vp.spectral_bc_mode = SpectralBCMode::SlipWall;
    add_rhs_viscous(U, g, eos, vp, Rhs);

    const Real k0 = M_PI / g.lx;   // 1.0
    const Real k4 = std::pow(k0, 4);
    Real l2_err = 0.0, l2_ref = 0.0;
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const Real x = g.xc(i);
                const Real expected =
                    -vp.hyper_coeff * k4 * A * std::cos(M_PI * x / g.lx);
                const Real got = Rhs[RHO](i, j, k);
                l2_err += (got - expected) * (got - expected);
                l2_ref += expected * expected;
            }
    // R2R round-trip + spectral multiplier roundoff is slightly worse than
    // r2c/c2r at this size; still many orders below the FD tolerance.
    EXPECT_LT(std::sqrt(l2_err / l2_ref), 1e-9);
}

// Wall-normal momentum on the x-axis uses DST-II (odd reflection).
// Lowest sine mode has k = pi/L = 1 for L = pi.
TEST(Hyperdissipation, SpectralSlipWallNabla4OnWallNormalMomentum) {
    const int N = 32;
    Grid g; g.nx = g.ny = g.nz = N;
    g.lx = g.ly = g.lz = M_PI;
    g.x0 = g.y0 = g.z0 = 0.0;

    IdealGas eos{GammaLaw{}};
    State U(N, N, N);
    const Real A = 0.1;

    // Set conserved variables directly. The hyperdissipation operator is
    // linear and acts on each conserved variable independently, so the
    // values don't need to form a physically consistent Euler state -- we
    // only check RHS[RHOU].
    for (int v = 0; v < NCONS; ++v) U.fill(v, 0.0);
    for (int k = -U.ng(); k < g.nz + U.ng(); ++k)
        for (int j = -U.ng(); j < g.ny + U.ng(); ++j)
            for (int i = -U.ng(); i < g.nx + U.ng(); ++i) {
                U[RHO ](i, j, k) = 1.0;
                U[RHOE](i, j, k) = 1.0;
                const Real x = g.xc(i);
                U[RHOU](i, j, k) = A * std::sin(M_PI * x / g.lx);
            }

    BCSet bc;
    bc.xlo = bc.xhi = BCType::SlipWall;
    bc.ylo = bc.yhi = BCType::SlipWall;
    bc.zlo = bc.zhi = BCType::SlipWall;
    apply_bcs(U, bc);

    State Rhs(N, N, N);
    for (int v = 0; v < NCONS; ++v) Rhs.fill(v, 0.0);

    ViscousParams vp;
    vp.mu = 0.0;
    vp.hyper_coeff = 0.01;
    vp.hyper6_coeff = 0.0;
    vp.hyper_method = HyperMethod::Pseudospectral;
    vp.spectral_bc_mode = SpectralBCMode::SlipWall;
    add_rhs_viscous(U, g, eos, vp, Rhs);

    const Real k0 = M_PI / g.lx;
    const Real k4 = std::pow(k0, 4);
    Real l2_err = 0.0, l2_ref = 0.0;
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const Real x = g.xc(i);
                const Real expected =
                    -vp.hyper_coeff * k4 * A * std::sin(M_PI * x / g.lx);
                const Real got = Rhs[RHOU](i, j, k);
                l2_err += (got - expected) * (got - expected);
                l2_ref += expected * expected;
            }
    EXPECT_LT(std::sqrt(l2_err / l2_ref), 1e-10);
}

TEST(Hyperdissipation, SpectralSlipWallNabla6OnDensityMode) {
    const int N = 32;
    Grid g; g.nx = g.ny = g.nz = N;
    g.lx = g.ly = g.lz = M_PI;
    g.x0 = g.y0 = g.z0 = 0.0;

    IdealGas eos{GammaLaw{}};
    State U(N, N, N);
    const Real rho_0 = 1.0, A = 0.1, p_0 = 1.0;
    for (int k = -U.ng(); k < g.nz + U.ng(); ++k)
        for (int j = -U.ng(); j < g.ny + U.ng(); ++j)
            for (int i = -U.ng(); i < g.nx + U.ng(); ++i) {
                const Real x = g.xc(i);
                const Real rho = rho_0 + A * std::cos(M_PI * x / g.lx);
                set_from_primitive(U, i, j, k, eos, rho, 0.0, 0.0, 0.0, p_0);
            }

    BCSet bc;
    bc.xlo = bc.xhi = BCType::SlipWall;
    bc.ylo = bc.yhi = BCType::SlipWall;
    bc.zlo = bc.zhi = BCType::SlipWall;
    apply_bcs(U, bc);

    State Rhs(N, N, N);
    for (int v = 0; v < NCONS; ++v) Rhs.fill(v, 0.0);

    ViscousParams vp;
    vp.mu = 0.0;
    vp.hyper_coeff = 0.0;
    vp.hyper6_coeff = 0.01;
    vp.hyper_method = HyperMethod::Pseudospectral;
    vp.spectral_bc_mode = SpectralBCMode::SlipWall;
    add_rhs_viscous(U, g, eos, vp, Rhs);

    const Real k0 = M_PI / g.lx;
    const Real k6 = std::pow(k0, 6);
    Real l2_err = 0.0, l2_ref = 0.0;
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const Real x = g.xc(i);
                const Real expected =
                    -vp.hyper6_coeff * k6 * A * std::cos(M_PI * x / g.lx);
                const Real got = Rhs[RHO](i, j, k);
                l2_err += (got - expected) * (got - expected);
                l2_ref += expected * expected;
            }
    // Same |k|^6 vs |k|^4 amplification rationale as the periodic test.
    EXPECT_LT(std::sqrt(l2_err / l2_ref), 1e-6);
}

TEST(Hyperdissipation, ZeroCoefficientHasNoEffect) {
    const int N = 16;
    Grid g; g.nx = g.ny = g.nz = N;
    g.lx = g.ly = g.lz = 2.0 * M_PI;
    g.x0 = g.y0 = g.z0 = 0.0;
    IdealGas eos{GammaLaw{}};
    State U(N, N, N);
    fill_density_mode(U, g, eos, 1.0, 0.1, 1.0, 1.0);

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;
    apply_bcs(U, bc);

    State Rhs(N, N, N);
    for (int v = 0; v < NCONS; ++v) Rhs.fill(v, 0.0);

    ViscousParams vp;
    vp.mu = 0.0;
    vp.hyper_coeff = 0.0;
    vp.hyper6_coeff = 0.0;
    add_rhs_viscous(U, g, eos, vp, Rhs);

    Real max_abs = 0.0;
    for (int v = 0; v < NCONS; ++v)
        for (int k = 0; k < N; ++k)
            for (int j = 0; j < N; ++j)
                for (int i = 0; i < N; ++i)
                    max_abs = std::max(max_abs, std::fabs(Rhs[v](i, j, k)));
    EXPECT_LT(max_abs, 1e-30);
}
