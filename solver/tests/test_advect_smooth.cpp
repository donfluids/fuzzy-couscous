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

Real run_advection(int N, Real t_end) {
    Grid g;
    g.nx = N; g.ny = 4; g.nz = 4;
    g.lx = 1.0; g.ly = g.lx / N * 4; g.lz = g.ly;
    g.x0 = 0.0; g.y0 = 0.0; g.z0 = 0.0;

    IdealGas eos{GammaLaw{}};
    State U(g.nx, g.ny, g.nz);
    const Real u0 = 1.0;
    ic_density_wave_x(U, g, eos, /*amp=*/0.1, /*kwave=*/1.0, u0);

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;

    RK3 driver(g.nx, g.ny, g.nz, U.ng());

    Real t = 0.0;
    const Real dt = 0.2 / N;
    while (t + dt < t_end) {
        driver.step(U, g, bc, eos, dt);
        t += dt;
    }
    if (t < t_end) driver.step(U, g, bc, eos, t_end - t);

    // Expected: density wave advected by u0 over time t_end.
    Real l2 = 0.0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real x = g.xc(i);
                const Real phase = 2.0 * M_PI * (x - u0 * t_end);
                const Real exact = 1.0 + 0.1 * std::sin(phase);
                const Real e = U[RHO](i, j, k) - exact;
                l2 += e * e;
            }
    return std::sqrt(l2 / (g.nx * g.ny * g.nz));
}

}  // namespace

// With dt ~ dx (fixed CFL), total error is dominated by temporal RK3 (~dt^3).
// To probe spatial order we'd need dt ~ dx^2, which is too expensive here.
TEST(AdvectSmooth, ConvergesAtRK3Order) {
    const Real t_end = 0.1;
    const Real e64  = run_advection(64,  t_end);
    const Real e128 = run_advection(128, t_end);
    const Real rate = std::log2(e64 / e128);
    EXPECT_GT(rate, 2.7);
    EXPECT_LT(rate, 3.3);
    EXPECT_LT(e128, 1e-3);
}
