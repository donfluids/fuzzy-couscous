#include <gtest/gtest.h>

#include "core/Grid.hpp"
#include "core/State.hpp"
#include "ic/Canonical.hpp"
#include "physics/EOS.hpp"

#include <cmath>

using namespace blast;

namespace {
Grid cube(int N) {
    Grid g;
    g.nx = N; g.ny = N; g.nz = N;
    g.lx = 1.0; g.ly = 1.0; g.lz = 1.0;
    g.x0 = -0.5; g.y0 = -0.5; g.z0 = -0.5;
    return g;
}
}  // namespace

// The Gaussian deposition must put exactly E_total of internal energy above
// ambient into the field (continuum normalization; sigma << box so truncation
// is negligible), keep density/pressure positive, and peak at the centre.
TEST(GaussianBlast, ConservesEnergyAndPositive) {
    Grid g = cube(64);
    GammaLaw gl; gl.gamma = 1.4; gl.R = 1.0;
    IdealGas eos{gl};
    State U(g.nx, g.ny, g.nz);

    const Real E_total = 10.0, sigma = 0.1, rho0 = 1.0, T0 = 1.0;
    ic_gaussian_blast_3d(U, g, eos, E_total, sigma, rho0, T0, /*Y42*/ 0.0);

    const Real p_amb = rho0 * gl.R * T0;
    const Real rhoE_amb = p_amb / (gl.gamma - 1.0);   // u = 0
    const Real dV = g.dx() * g.dy() * g.dz();

    Real deposited = 0.0, rho_min = 1e30, p_min = 1e30;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real rho = U[RHO](i, j, k);
                const Real u = U[RHOU](i, j, k) / rho;
                const Real p = eos.pressure(rho, U[RHOE](i, j, k) - 0.5 * rho * u * u);
                deposited += (U[RHOE](i, j, k) - rhoE_amb) * dV;
                rho_min = std::min(rho_min, rho);
                p_min = std::min(p_min, p);
            }

    EXPECT_NEAR(deposited, E_total, 0.02 * E_total) << "deposited energy != E_total";
    EXPECT_GT(rho_min, 0.0);
    EXPECT_GT(p_min, 0.0);
    EXPECT_DOUBLE_EQ(rho_min, rho0) << "density must be uniform (no contact)";

    const int c = g.nx / 2;
    const Real p_center = eos.pressure(U[RHO](c, c, c), U[RHOE](c, c, c));
    EXPECT_GT(p_center, 10.0 * p_amb) << "pressure should peak at the centre";
}

// C-infinity smoothness: the cell-to-cell pressure ratio across the steepest
// part of a resolved Gaussian (sigma ~ 6 cells) must stay mild -- no
// discontinuity for the central/AFP scheme to overshoot on. Contrast with a
// tanh hot-sphere at ~1.5 dx, which jumps by orders of magnitude in one cell.
TEST(GaussianBlast, SmoothNoJump) {
    Grid g = cube(64);
    GammaLaw gl; gl.gamma = 1.4; gl.R = 1.0;
    IdealGas eos{gl};
    State U(g.nx, g.ny, g.nz);
    ic_gaussian_blast_3d(U, g, eos, 10.0, 0.1, 1.0, 1.0, 0.0);

    const int jc = g.ny / 2, kc = g.nz / 2;
    Real max_ratio = 1.0;
    for (int i = 1; i < g.nx; ++i) {
        const Real p0 = eos.pressure(U[RHO](i - 1, jc, kc), U[RHOE](i - 1, jc, kc));
        const Real p1 = eos.pressure(U[RHO](i, jc, kc), U[RHOE](i, jc, kc));
        max_ratio = std::max(max_ratio, std::max(p0 / p1, p1 / p0));
    }
    EXPECT_LT(max_ratio, 2.0) << "adjacent-cell pressure ratio too large (not smooth)";
}
