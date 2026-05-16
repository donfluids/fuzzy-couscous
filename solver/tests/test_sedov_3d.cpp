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
#include <vector>

using namespace blast;

namespace {

// Sedov-Taylor 3D shock radius:  R(t) = xi_0 (E t^2 / rho_0)^(1/5).
// xi_0 = 1.0328 for gamma = 1.4 (Sedov 1959; Landau-Lifshitz Vol 6 sec. 106).
Real sedov_radius(Real gamma, Real E, Real rho0, Real t) {
    (void)gamma;  // we only validate at gamma = 1.4
    constexpr Real xi0 = 1.0328;
    return xi0 * std::pow(E * t * t / rho0, 0.2);
}

// Radial-bin the density field and return the radius of the bin with the
// largest mean density (i.e. the shock).
Real measure_shock_radius(const State& U, const Grid& g, Real rho_ambient) {
    const Real xc = g.x0 + 0.5 * g.lx;
    const Real yc = g.y0 + 0.5 * g.ly;
    const Real zc = g.z0 + 0.5 * g.lz;
    const Real bin_w = g.dx();          // one cell wide
    const int n_bins = static_cast<int>(0.5 * g.lx / bin_w);

    std::vector<Real> rho_sum(n_bins, 0.0);
    std::vector<int>  count(n_bins, 0);

    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real xv = g.xc(i) - xc;
                const Real yv = g.yc(j) - yc;
                const Real zv = g.zc(k) - zc;
                const Real r  = std::sqrt(xv*xv + yv*yv + zv*zv);
                int b = static_cast<int>(r / bin_w);
                if (b >= 0 && b < n_bins) {
                    rho_sum[b] += U[RHO](i, j, k);
                    count[b]++;
                }
            }

    int best = -1;
    Real best_rho = rho_ambient;
    for (int b = 0; b < n_bins; ++b) {
        if (count[b] < 8) continue;
        const Real mean = rho_sum[b] / count[b];
        if (mean > best_rho) {
            best_rho = mean;
            best = b;
        }
    }
    return (best >= 0) ? (best + 0.5) * bin_w : 0.0;
}

}  // namespace

TEST(Sedov3D, ShockRadiusMatchesSelfSimilar) {
    Grid g;
    g.nx = g.ny = g.nz = 64;
    g.lx = g.ly = g.lz = 1.2;
    g.x0 = g.y0 = g.z0 = -0.6;

    IdealGas eos{GammaLaw{}};
    State U(g.nx, g.ny, g.nz);

    const Real E       = 1.0;
    const Real rho0    = 1.0;
    const Real p_amb   = 1e-5;
    const Real r_blast = 4.0 * g.dx();   // standard: ~4 cells

    ic_sedov_3d(U, g, eos, E, rho0, p_amb, r_blast);

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Outflow;
    bc.ylo = bc.yhi = BCType::Outflow;
    bc.zlo = bc.zhi = BCType::Outflow;

    RK3 driver(g.nx, g.ny, g.nz, U.ng());

    const Real t_end = 0.05;
    Real t = 0.0;
    int steps = 0;
    while (t < t_end && steps < 10000) {
        Real dt = max_dt_hyperbolic(U, g, eos, 0.3);
        ASSERT_TRUE(std::isfinite(dt) && dt > 0.0)
            << "non-finite dt at step " << steps << " t=" << t;
        if (t + dt > t_end) dt = t_end - t;
        driver.step(U, g, bc, eos, dt);
        t += dt;
        ++steps;
    }

    const Real R_measured = measure_shock_radius(U, g, rho0);
    const Real R_analytic = sedov_radius(eos.eos.gamma, E, rho0, t_end);

    const Real rel_err = std::fabs(R_measured - R_analytic) / R_analytic;

    // Bin resolution is dx; on a 64-cube domain that's 1.875%. Energy
    // deposition spreads E over a few cells, which dilates the early-time
    // radius slightly; total relative error budget here is 5%.
    EXPECT_LT(rel_err, 0.05) << "R_measured=" << R_measured
                             << " R_analytic=" << R_analytic
                             << " steps=" << steps;
}
