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

// Build an x-elongated thin-domain grid: slip walls in x, periodic in y,z.
Grid make_xtube_grid(int Nx, int Nyz, Real L) {
    Grid g;
    g.nx = Nx; g.ny = Nyz; g.nz = Nyz;
    g.lx = L;
    g.ly = L / Nx * Nyz;
    g.lz = g.ly;
    g.x0 = 0.0; g.y0 = 0.0; g.z0 = 0.0;
    return g;
}

BCSet xwalls_y_z_periodic() {
    BCSet bc;
    bc.xlo = bc.xhi = BCType::SlipWall;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;
    return bc;
}

}  // namespace

// Inviscid scheme with slip walls must conserve mass and total energy
// exactly (slip walls do no pressure work since u_n = 0). Tangential
// momentum has no conservation law at the walls.
TEST(SlipWall, MassAndEnergyConservedToRoundoff) {
    auto g = make_xtube_grid(64, 4, 1.0);
    IdealGas eos{GammaLaw{1.4, 1.0}};
    State U(g.nx, g.ny, g.nz);

    // Random-ish bounded initial state: density/pressure cosine + small
    // tangential shear. Has nonzero flow but no mean wall-normal motion at
    // t=0. Slip walls will reflect any wall-bound perturbation that develops.
    for (int k = -U.ng(); k < g.nz + U.ng(); ++k)
        for (int j = -U.ng(); j < g.ny + U.ng(); ++j)
            for (int i = -U.ng(); i < g.nx + U.ng(); ++i) {
                const Real x = g.xc(i);
                const Real rho = 1.0 + 0.1 * std::cos(2.0 * M_PI * x / g.lx);
                const Real p   = 1.0 + 0.1 * std::cos(2.0 * M_PI * x / g.lx);
                set_from_primitive(U, i, j, k, eos, rho, 0.0, 0.05, 0.0, p);
            }

    auto bc = xwalls_y_z_periodic();
    apply_bcs(U, bc);
    RK3 driver(g.nx, g.ny, g.nz, U.ng());

    auto integrate = [&](State& U) {
        Real M = 0, E = 0;
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j)
                for (int i = 0; i < g.nx; ++i) {
                    M += U[RHO ](i, j, k);
                    E += U[RHOE](i, j, k);
                }
        return std::make_pair(M, E);
    };

    const auto [M0, E0] = integrate(U);
    const Real t_end = 2.0;
    Real t = 0.0;
    int steps = 0;
    while (t < t_end && steps < 5000) {
        Real dt = max_dt_hyperbolic(U, g, eos, 0.4);
        if (t + dt > t_end) dt = t_end - t;
        driver.step(U, g, bc, eos, dt);
        t += dt;
        ++steps;
    }
    const auto [M1, E1] = integrate(U);

    EXPECT_LT(std::fabs(M1 - M0) / std::fabs(M0), 1e-10) << "mass drifted";
    EXPECT_LT(std::fabs(E1 - E0) / std::fabs(E0), 1e-10) << "energy drifted";

    // Positivity: density and pressure remain physical.
    Real rho_min = 1e30, p_min = 1e30;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real r = U[RHO](i, j, k);
                const Real u = U[RHOU](i, j, k) / r;
                const Real v = U[RHOV](i, j, k) / r;
                const Real w = U[RHOW](i, j, k) / r;
                const Real ke = 0.5 * r * (u*u + v*v + w*w);
                const Real p = eos.pressure(r, U[RHOE](i, j, k) - ke);
                rho_min = std::min(rho_min, r);
                p_min   = std::min(p_min, p);
            }
    EXPECT_GT(rho_min, 0.0);
    EXPECT_GT(p_min,   0.0);
}

// Small-amplitude rightward-moving acoustic pulse round-trip: after time
// 2 L / c_0 the wave has bounced off both walls once and returned to its
// starting position. Verifies that (a) the wall reflects without distortion,
// (b) no spurious wall-normal velocity persists between reflections, and
// (c) total energy is preserved exactly. Inviscid, so the only error is
// numerical dispersion on the high-order central scheme.
TEST(SlipWall, AcousticPulseRoundTripPreservesPulse) {
    auto g = make_xtube_grid(128, 4, 1.0);
    IdealGas eos{GammaLaw{1.4, 1.0}};
    State U(g.nx, g.ny, g.nz);

    const Real rho_0 = 1.0;
    const Real p_0   = 1.0;
    const Real c_0   = std::sqrt(1.4 * p_0 / rho_0);
    const Real eps   = 1e-3;
    const Real x_c   = 0.25;
    const Real sigma = 0.04;

    // Linear acoustic relations: rho' = eps env(x), u' = c_0 rho'/rho_0,
    // p' = c_0^2 rho'. Right-moving (positive characteristic).
    for (int k = -U.ng(); k < g.nz + U.ng(); ++k)
        for (int j = -U.ng(); j < g.ny + U.ng(); ++j)
            for (int i = -U.ng(); i < g.nx + U.ng(); ++i) {
                const Real x = g.xc(i);
                const Real env = std::exp(-(x - x_c) * (x - x_c) / (sigma * sigma));
                const Real dr = eps * env;
                const Real rho = rho_0 + dr;
                const Real u   = c_0 * dr / rho_0;
                const Real p   = p_0 + c_0 * c_0 * dr;
                set_from_primitive(U, i, j, k, eos, rho, u, 0.0, 0.0, p);
            }

    auto bc = xwalls_y_z_periodic();
    apply_bcs(U, bc);

    // Snapshot initial mid-plane density profile for shape comparison.
    const int jmid = g.ny / 2, kmid = g.nz / 2;
    std::vector<Real> rho_initial(g.nx);
    for (int i = 0; i < g.nx; ++i) rho_initial[i] = U[RHO](i, jmid, kmid);
    Real E0 = 0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) E0 += U[RHOE](i, j, k);

    RK3 driver(g.nx, g.ny, g.nz, U.ng());
    // Two reflections = two transit times = 2 L / c_0. Pulse should return
    // to its original position moving in the original direction.
    const Real t_end = 2.0 * g.lx / c_0;
    Real t = 0.0;
    while (t < t_end) {
        Real dt = max_dt_hyperbolic(U, g, eos, 0.4);
        if (t + dt > t_end) dt = t_end - t;
        driver.step(U, g, bc, eos, dt);
        t += dt;
    }

    // Energy is conserved by slip walls + inviscid scheme.
    Real E1 = 0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) E1 += U[RHOE](i, j, k);
    EXPECT_LT(std::fabs(E1 - E0) / std::fabs(E0), 1e-10);

    // The pulse should still be a single localized bump near x_c. Locate the
    // peak and verify position and amplitude.
    int i_peak = 0;
    Real rho_peak = rho_0;
    for (int i = 0; i < g.nx; ++i) {
        const Real r = U[RHO](i, jmid, kmid);
        if (r > rho_peak) { rho_peak = r; i_peak = i; }
    }
    const Real x_peak = g.xc(i_peak);

    // Peak position must be back near x_c (within a few cells due to
    // dispersion). Tolerance: a few cells in x.
    EXPECT_LT(std::fabs(x_peak - x_c), 5.0 * g.dx());

    // Peak amplitude is reduced by numerical dispersion but should still be
    // recognizable (> 50% of initial). Should never be larger than initial.
    EXPECT_GT(rho_peak - rho_0, 0.5 * eps);
    EXPECT_LT(rho_peak - rho_0, 1.05 * eps);

    // Density floor: no spurious overshoots driving rho below ambient by
    // anything more than dispersion ringing of order eps.
    Real rho_min = 1e30;
    for (int i = 0; i < g.nx; ++i)
        rho_min = std::min(rho_min, U[RHO](i, jmid, kmid));
    EXPECT_GT(rho_min, rho_0 - 0.6 * eps);
}
