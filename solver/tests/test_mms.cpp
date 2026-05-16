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
#include <iostream>

using namespace blast;

namespace {

// Map a difference d into the nearest-image range [-L/2, L/2).
Real nearest_image(Real d, Real L) {
    d -= L * std::floor(d / L + 0.5);
    return d;
}

// MMS-A run: 3D entropy wave on a periodic cube of side 2 pi. Carrier wave
// velocity u0=v0=w0=1 so the analytic solution returns to its initial state
// at t = 2 pi (one full traversal in each direction). mu = 0 (inviscid).
// Returns L2(rho_final - rho_analytic), normalized by sqrt(N^3).
Real run_entropy_wave(int N) {
    Grid g;
    g.nx = g.ny = g.nz = N;
    g.lx = g.ly = g.lz = 2.0 * M_PI;
    g.x0 = g.y0 = g.z0 = 0.0;

    IdealGas eos{GammaLaw{}};
    State U(g.nx, g.ny, g.nz);

    const Real A = 0.1, u0 = 1.0, v0 = 1.0, w0 = 1.0;
    const Real rho_0 = 1.0, p_0 = 1.0;
    ic_entropy_wave_3d(U, g, eos, A, u0, v0, w0, rho_0, p_0);

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;
    apply_bcs(U, bc);

    RK3 driver(g.nx, g.ny, g.nz, U.ng());

    const Real t_end = 2.0 * M_PI;     // one full periodic traversal
    Real t = 0.0;
    while (t < t_end) {
        Real dt = max_dt_hyperbolic(U, g, eos, 0.4);
        if (t + dt > t_end) dt = t_end - t;
        driver.step(U, g, bc, eos, dt);
        t += dt;
    }

    // Analytic rho at t_end. With u0 = v0 = w0 = 1 and L = 2 pi, the wave
    // has traversed the domain exactly once, so analytic equals the IC.
    const Real kx = 2.0 * M_PI / g.lx;
    const Real ky = 2.0 * M_PI / g.ly;
    const Real kz = 2.0 * M_PI / g.lz;
    Real l2 = 0.0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real x = g.xc(i), y = g.yc(j), z = g.zc(k);
                const Real phix = kx * (x - u0 * t_end);
                const Real phiy = ky * (y - v0 * t_end);
                const Real phiz = kz * (z - w0 * t_end);
                const Real rho_an = rho_0 * (1.0 + A * std::sin(phix)
                                                    * std::sin(phiy)
                                                    * std::sin(phiz));
                const Real e = U[RHO](i, j, k) - rho_an;
                l2 += e * e;
            }
    return std::sqrt(l2 / (static_cast<Real>(g.nx) * g.ny * g.nz));
}

// MMS-B run: 2D Yee-Sandham-Djomehri isentropic Euler vortex of strength
// eps=5 on a cubic periodic domain of side L=10. Vortex centered at (5,5)
// at t=0, translating at (u_inf, v_inf) = (1, 1). At t = L the analytic
// vortex center has returned to (5, 5); analytic state = initial state.
struct VortexErrs { Real rho, ux, uy; };
VortexErrs run_vortex(int N) {
    Grid g;
    g.nx = g.ny = g.nz = N;
    g.lx = g.ly = g.lz = 10.0;
    g.x0 = g.y0 = g.z0 = 0.0;

    GammaLaw gl; gl.gamma = 1.4; gl.R = 1.0;
    IdealGas eos{gl};
    State U(g.nx, g.ny, g.nz);

    const Real eps = 5.0;
    const Real u_inf = 1.0, v_inf = 1.0;
    const Real x_c0 = 5.0, y_c0 = 5.0;
    ic_isentropic_vortex(U, g, eos, eps, u_inf, v_inf, x_c0, y_c0);

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;
    apply_bcs(U, bc);

    RK3 driver(g.nx, g.ny, g.nz, U.ng());

    const Real t_end = g.lx / u_inf;   // one full traversal -> back to start
    Real t = 0.0;
    while (t < t_end) {
        Real dt = max_dt_hyperbolic(U, g, eos, 0.4);
        if (t + dt > t_end) dt = t_end - t;
        driver.step(U, g, bc, eos, dt);
        t += dt;
    }

    const Real coeff   = (gl.gamma - 1.0) * eps * eps / (8.0 * gl.gamma * M_PI * M_PI);
    const Real two_pi_inv = 1.0 / (2.0 * M_PI);

    Real l2_rho = 0, l2_ux = 0, l2_uy = 0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real x = g.xc(i), y = g.yc(j);
                const Real xd = nearest_image(x - x_c0, g.lx);
                const Real yd = nearest_image(y - y_c0, g.ly);
                const Real r2 = xd * xd + yd * yd;
                const Real e_factor = std::exp((1.0 - r2) * 0.5);
                const Real T = 1.0 - coeff * std::exp(1.0 - r2);
                const Real rho_an = std::pow(T, 1.0 / (gl.gamma - 1.0));
                const Real u_an   = u_inf - eps * yd * two_pi_inv * e_factor;
                const Real v_an   = v_inf + eps * xd * two_pi_inv * e_factor;

                const Real rho = U[RHO ](i, j, k);
                const Real ux  = U[RHOU](i, j, k) / rho;
                const Real uy  = U[RHOV](i, j, k) / rho;

                l2_rho += (rho - rho_an) * (rho - rho_an);
                l2_ux  += (ux - u_an)   * (ux - u_an);
                l2_uy  += (uy - v_an)   * (uy - v_an);
            }
    const Real norm = 1.0 / (static_cast<Real>(g.nx) * g.ny * g.nz);
    return {std::sqrt(l2_rho * norm), std::sqrt(l2_ux * norm),
            std::sqrt(l2_uy * norm)};
}

}  // namespace

TEST(MMS3D, EntropyWaveConvergesAtRK3Order) {
    const Real e32 = run_entropy_wave(32);
    const Real e64 = run_entropy_wave(64);
    const Real rate = std::log2(e32 / e64);
    std::cerr << "[MMS-A entropy wave] e32=" << e32
              << " e64=" << e64 << " rate=" << rate << "\n";
    EXPECT_GT(rate, 2.7) << "expected ~3.0 (RK3 third-order)";
    EXPECT_LT(e64, 1e-3) << "absolute error at 64^3 too large";
}

TEST(MMS3D, IsentropicVortexConvergesAtRK3Order) {
    const VortexErrs e32 = run_vortex(32);
    const VortexErrs e64 = run_vortex(64);
    const Real rate_rho = std::log2(e32.rho / e64.rho);
    const Real rate_ux  = std::log2(e32.ux  / e64.ux );
    const Real rate_uy  = std::log2(e32.uy  / e64.uy );
    std::cerr << "[MMS-B vortex] rho rate=" << rate_rho
              << " ux rate=" << rate_ux << " uy rate=" << rate_uy
              << " (e64 rho=" << e64.rho << ")\n";
    // Vortex on 32^3 in L=10 has only ~3 cells per vortex scale; allow a
    // slightly relaxed threshold but require monotonic improvement.
    EXPECT_GT(rate_rho, 2.5) << "rho convergence too shallow";
    EXPECT_GT(rate_ux,  2.5) << "u_x convergence too shallow";
    EXPECT_GT(rate_uy,  2.5) << "u_y convergence too shallow";
}
