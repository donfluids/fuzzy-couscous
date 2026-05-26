#include <gtest/gtest.h>

#include "bc/BC.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "ic/Canonical.hpp"
#include "numerics/RK3.hpp"
#include "physics/EOS.hpp"

#include <cmath>
#include <iostream>

using namespace blast;

namespace {

// Smooth periodic entropy wave: rho = 1 + A sin(2 pi x), u = u0, p = 1
// (single ideal gas). Velocity and pressure stay uniform to round-off, so the
// Ducros sensor never trips and the inviscid path runs in pure 6th-order
// central mode. dt is scaled like dx^2 so the SSP-RK3 temporal error
// O(dt^3) = O(dx^6) decays at the same rate as the spatial error, letting the
// measured convergence reveal the 6th-order SPATIAL accuracy of the assembled
// flux divergence + RK3 + periodic BCs.
Real run_entropy_wave_L2(int N) {
    Grid g;
    g.nx = N; g.ny = 4; g.nz = 4;
    g.lx = 1.0; g.ly = g.lx / N * 4; g.lz = g.ly;
    g.x0 = 0.0; g.y0 = 0.0; g.z0 = 0.0;

    IdealGas eos{GammaLaw{}};
    State U(g.nx, g.ny, g.nz);
    const Real A = 0.1, u0 = 1.0;
    ic_density_wave_x(U, g, eos, A, /*kwave=*/1.0, u0);

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;

    RK3 driver(g.nx, g.ny, g.nz, U.ng());

    // dt = K dx^2 with K = 2 → CFL ≈ (|u|+c)·K·dx ≈ 0.27 on N=16, smaller on
    // finer grids (stable for RK3 + 6th-order central). nsteps chosen so the
    // run lands exactly on t_end with dt ≈ K dx^2.
    const Real t_end = 0.1;
    const Real dx = g.lx / N;
    const Real K = 2.0;
    const int nsteps = std::max(1, (int)std::lround(t_end / (K * dx * dx)));
    const Real dt = t_end / nsteps;

    for (int s = 0; s < nsteps; ++s) driver.step(U, g, bc, eos, dt);

    Real l2 = 0.0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real x = g.xc(i);
                const Real exact = 1.0 + A * std::sin(2.0 * M_PI * (x - u0 * t_end));
                const Real e = U[RHO](i, j, k) - exact;
                l2 += e * e;
            }
    return std::sqrt(l2 / (g.nx * g.ny * g.nz));
}

}  // namespace

// With dt ~ dx^2 the temporal error O(dt^3) tracks the spatial error O(dx^6),
// so the asymptotic rate exposes the central6 reconstruction. WENO5 (if the
// sensor fired) would cap the rate near 5; central6 lands near 6.
TEST(OrderSmooth, SixthOrderInviscidCentral) {
    const Real e16 = run_entropy_wave_L2(16);
    const Real e32 = run_entropy_wave_L2(32);
    const Real e64 = run_entropy_wave_L2(64);
    const Real r_1632 = std::log2(e16 / e32);
    const Real r_3264 = std::log2(e32 / e64);
    std::cerr << "[OrderSmooth] e16=" << e16 << " e32=" << e32 << " e64=" << e64
              << " rate(16->32)=" << r_1632
              << " rate(32->64)=" << r_3264 << "\n";
    EXPECT_GT(r_3264, 5.5) << "asymptotic spatial order should approach 6";
    EXPECT_GT(r_1632, 5.0);
}
