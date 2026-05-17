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
    add_rhs_viscous(U, g, eos, vp, Rhs);

    Real max_abs = 0.0;
    for (int v = 0; v < NCONS; ++v)
        for (int k = 0; k < N; ++k)
            for (int j = 0; j < N; ++j)
                for (int i = 0; i < N; ++i)
                    max_abs = std::max(max_abs, std::fabs(Rhs[v](i, j, k)));
    EXPECT_LT(max_abs, 1e-30);
}
