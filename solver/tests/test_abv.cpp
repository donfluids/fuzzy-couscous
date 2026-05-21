#include <gtest/gtest.h>

#include "bc/BC.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "ic/Canonical.hpp"
#include "numerics/ArtificialDiffusivity.hpp"
#include "numerics/Gradients.hpp"
#include "numerics/RHS.hpp"
#include "numerics/RK3.hpp"
#include "physics/EOS.hpp"
#include "physics/RiemannExact.hpp"

#include <algorithm>
#include <cmath>

using namespace blast;

namespace {

ViscousParams lad_params() {
    ViscousParams vp;
    vp.mu = 0.0;
    vp.abv_enabled = true;
    vp.abv_cbeta = 1.0;
    vp.abv_cmu = 0.002;
    vp.abv_ckappa = 0.01;
    vp.abv_disable_weno = true;
    return vp;
}

Grid thin_grid(int N) {
    Grid g;
    g.nx = N; g.ny = 4; g.nz = 4;
    g.lx = 1.0; g.ly = g.lx / N * 4; g.lz = g.ly;
    g.x0 = 0.0; g.y0 = 0.0; g.z0 = 0.0;
    return g;
}

// Fill a uniform-density, uniform-pressure field with a 1D x-velocity profile
// u(x) = amp * tanh((x - 0.5)/delta). amp < 0 makes a compression at x=0.5
// (du/dx < 0); amp > 0 makes an expansion there.
void set_tanh_velocity(State& U, const Grid& g, const IdealGas& eos, Real amp) {
    const Real xc = 0.5, delta = 2.0 * g.dx();
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real x = g.xc(i);
                const Real u = amp * std::tanh((x - xc) / delta);
                const Real rho = 1.0, p = 1.0;
                U[RHO](i, j, k)  = rho;
                U[RHOU](i, j, k) = rho * u;
                U[RHOV](i, j, k) = 0.0;
                U[RHOW](i, j, k) = 0.0;
                U[RHOE](i, j, k) = p / (eos.eos.gamma - 1.0) + 0.5 * rho * u * u;
            }
}

struct LADFields {
    Field3D theta, strain, mu, beta, kappa;
    Real nu_max;
};

// Compute the LAD coefficient fields for a (ghost-filled) state.
LADFields eval_lad(const State& U, const Grid& g, const IdealGas& eos,
                   const ViscousParams& vp) {
    CellGradients G;
    G.allocate(g.nx, g.ny, g.nz, U.ng());
    Field3D pu, pv, pw, pT;
    LADFields f;
    for (Field3D* p : {&pu, &pv, &pw, &pT, &f.theta, &f.strain, &f.mu, &f.beta, &f.kappa})
        p->resize(g.nx, g.ny, g.nz, U.ng());
    compute_cell_gradients(U, g, eos, pu, pv, pw, pT, G);
    f.nu_max = compute_lad_fields(U, g, eos, vp, G, pT,
                                  f.theta, f.strain, f.mu, f.beta, f.kappa);
    return f;
}

BCSet shock_bc() {
    BCSet bc;
    bc.xlo = bc.xhi = BCType::Outflow;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;
    return bc;
}

}  // namespace

// Artificial bulk viscosity must concentrate at the compression and vanish in
// the smooth tails -- the whole point of a localized (vs. global) dissipation.
TEST(ABV, BulkLocalizesAtCompression) {
    Grid g = thin_grid(64);
    IdealGas eos{GammaLaw{}};
    State U(g.nx, g.ny, g.nz);
    set_tanh_velocity(U, g, eos, -1.0);   // compression at x=0.5
    BCSet bc = shock_bc();
    apply_bcs(U, bc);

    LADFields f = eval_lad(U, g, eos, lad_params());
    const int jc = g.ny / 2, kc = g.nz / 2, ic = g.nx / 2;

    Real beta_peak = 0.0;
    for (int i = ic - 3; i <= ic + 3; ++i) beta_peak = std::max(beta_peak, f.beta(i, jc, kc));
    const Real beta_tail = f.beta(4, jc, kc);

    EXPECT_GT(beta_peak, 0.0) << "bulk viscosity should fire at the compression";
    EXPECT_LT(beta_tail, 1e-3 * beta_peak) << "should vanish in the smooth tail";
    EXPECT_GT(f.nu_max, 0.0);
}

// The compression switch H(-div u) must zero the bulk viscosity in an
// expansion, even though the 2nd-derivative sensor is large there. The shear
// viscosity (no switch) should still fire on the strain.
TEST(ABV, BulkSwitchOffInExpansion) {
    Grid g = thin_grid(64);
    IdealGas eos{GammaLaw{}};
    State U(g.nx, g.ny, g.nz);
    set_tanh_velocity(U, g, eos, +1.0);   // expansion at x=0.5 (div u > 0)
    BCSet bc = shock_bc();
    apply_bcs(U, bc);

    LADFields f = eval_lad(U, g, eos, lad_params());
    const int jc = g.ny / 2, kc = g.nz / 2, ic = g.nx / 2;

    Real beta_peak = 0.0, mu_peak = 0.0;
    for (int i = ic - 3; i <= ic + 3; ++i) {
        beta_peak = std::max(beta_peak, f.beta(i, jc, kc));
        mu_peak   = std::max(mu_peak,   f.mu(i, jc, kc));
    }
    EXPECT_NEAR(beta_peak, 0.0, 1e-30) << "bulk viscosity must switch off in expansion";
    EXPECT_GT(mu_peak, 0.0) << "shear viscosity fires regardless of compression sign";
}

// Smooth flow: LAD must converge away rapidly under refinement (here we just
// require it to be negligible relative to a representative |div u| scale on a
// well-resolved smooth sinusoid).
TEST(ABV, NegligibleInSmoothFlow) {
    Grid g = thin_grid(64);
    IdealGas eos{GammaLaw{}};
    State U(g.nx, g.ny, g.nz);
    // Smooth, well-resolved compression wave: u = -0.1 sin(2 pi x).
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real u = -0.1 * std::sin(2.0 * M_PI * g.xc(i));
                U[RHO](i, j, k)  = 1.0;
                U[RHOU](i, j, k) = u;
                U[RHOV](i, j, k) = 0.0;
                U[RHOW](i, j, k) = 0.0;
                U[RHOE](i, j, k) = 1.0 / (eos.eos.gamma - 1.0) + 0.5 * u * u;
            }
    BCSet bc;
    bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;
    apply_bcs(U, bc);

    LADFields f = eval_lad(U, g, eos, lad_params());
    const int jc = g.ny / 2, kc = g.nz / 2;
    Real beta_max = 0.0;
    for (int i = 0; i < g.nx; ++i) beta_max = std::max(beta_max, f.beta(i, jc, kc));
    // beta ~ C_beta rho |D^2 theta| h^2, with theta = -0.2 pi cos(2 pi x). The
    // grid-scaled 2nd difference of a well-resolved wave is tiny; require beta
    // far below the physical scale rho * |div u|_max * h^2.
    const Real scale = 1.0 * (0.2 * M_PI) * (g.dx() * g.dx());
    EXPECT_LT(beta_max, 0.05 * scale) << "LAD must be negligible in smooth flow";
}

namespace {

// A finite-amplitude acoustic wave on a fully periodic box steepens into a
// shock. Returns the minimum density at t_end, or -1 if the run went
// non-finite. Periodic BCs fill every ghost (no corner-ghost issue), so this
// isolates LAD's shock-capturing behaviour. The actual production target
// (slip-wall blast) likewise fills corners; outflow-BC viscous runs are a
// separate, pre-existing limitation of the gradient pass.
Real run_periodic_steepening(bool abv_on, bool disable_weno, Real cbeta, Real cmu, Real ckappa, Real dtfac) {
    Grid g;
    g.nx = 64; g.ny = 4; g.nz = 4;
    g.lx = 1.0; g.ly = g.lx / g.nx * g.ny; g.lz = g.ly;
    g.x0 = 0.0; g.y0 = 0.0; g.z0 = 0.0;
    IdealGas eos{GammaLaw{}};
    State U(g.nx, g.ny, g.nz);
    const Real A = 0.8;   // M_max ~ 0.68; steepens into a shock by t ~ 0.2
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real u = A * std::sin(2.0 * M_PI * g.xc(i));
                U[RHO](i, j, k)  = 1.0;
                U[RHOU](i, j, k) = u;
                U[RHOV](i, j, k) = 0.0;
                U[RHOW](i, j, k) = 0.0;
                U[RHOE](i, j, k) = 1.0 / (eos.eos.gamma - 1.0) + 0.5 * u * u;
            }
    BCSet bc;  // all periodic (default)

    RK3 driver(g.nx, g.ny, g.nz, U.ng());
    ViscousParams vp = lad_params();
    vp.abv_enabled = abv_on;
    vp.abv_disable_weno = disable_weno;
    vp.abv_cbeta = cbeta; vp.abv_cmu = cmu; vp.abv_ckappa = ckappa;
    const Real t_end = 0.3;
    const Real dxm = std::min({g.dx(), g.dy(), g.dz()});

    Real t = 0.0;
    int steps = 0;
    while (t < t_end && steps < 200000) {
        Real dt = max_dt_hyperbolic(U, g, eos, 0.4);
        const Real num = driver.last_abv_nu_max();
        if (num > 0.0) dt = std::min(dt, dtfac * dxm * dxm / num);
        if (t + dt > t_end) dt = t_end - t;
        driver.step(U, g, bc, eos, vp, dt);
        t += dt;
        ++steps;
    }

    const int jc = g.ny / 2, kc = g.nz / 2;
    Real rho_min = 1e30;
    for (int i = 0; i < g.nx; ++i) {
        const Real rho = U[RHO](i, jc, kc);
        const Real u = U[RHOU](i, jc, kc) / rho;
        const Real p = eos.pressure(rho, U[RHOE](i, jc, kc) - 0.5 * rho * u * u);
        if (!std::isfinite(rho) || !std::isfinite(p)) return -1.0;
        rho_min = std::min(rho_min, rho);
    }
    return rho_min;
}

}  // namespace

// With zero coefficients, enabling LAD must be a no-op: identical result to the
// pure hybrid scheme (guards against the artificial path corrupting anything).
TEST(ABV, ZeroCoeffIsNoOp) {
    const Real weno = run_periodic_steepening(false, false, 0, 0, 0, 0.25);
    const Real lad0 = run_periodic_steepening(true, false, 0, 0, 0, 0.25);
    EXPECT_GT(weno, 0.0);
    EXPECT_DOUBLE_EQ(weno, lad0) << "zero-coefficient LAD changed the solution";
}

// LAD added to the hybrid central/WENO scheme must stay stable and positive
// through shock formation (the default-safe configuration).
TEST(ABV, PeriodicShockLADAdditive) {
    const Real rho_min = run_periodic_steepening(true, false, 1.0, 0.002, 0.01, 0.25);
    EXPECT_GT(rho_min, 0.0) << "LAD additive to WENO went non-finite / negative";
}

// LAD as the SOLE shock treatment (WENO suppressed): the central scheme + LAD
// must capture the self-steepening shock without blowing up. This is the
// "replace WENO with a controlled dissipation" demonstration.
TEST(ABV, PeriodicShockLADOnly) {
    const Real rho_min = run_periodic_steepening(true, true, 1.0, 0.002, 0.01, 0.25);
    EXPECT_GT(rho_min, 0.0)
        << "LAD-only failed to keep the steepening shock finite and positive";
}

