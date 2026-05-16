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
    const Real e48 = run_entropy_wave(48);
    const Real e64 = run_entropy_wave(64);
    const Real r_3248 = std::log2(e32 / e48) / std::log2(48.0 / 32.0);
    const Real r_4864 = std::log2(e48 / e64) / std::log2(64.0 / 48.0);
    std::cerr << "[MMS-A entropy wave] e32=" << e32
              << " e48=" << e48 << " e64=" << e64
              << " rate(32->48)=" << r_3248
              << " rate(48->64)=" << r_4864 << "\n";
    EXPECT_GT(r_3248, 2.7) << "expected ~3.0 (RK3 third-order)";
    EXPECT_GT(r_4864, 2.7);
    EXPECT_LT(e64, 1e-3);
}

TEST(MMS3D, IsentropicVortexConvergesAtRK3Order) {
    const VortexErrs e32 = run_vortex(32);
    const VortexErrs e48 = run_vortex(48);
    const VortexErrs e64 = run_vortex(64);
    const Real r_3248 = std::log2(e32.rho / e48.rho) / std::log2(48.0 / 32.0);
    const Real r_4864 = std::log2(e48.rho / e64.rho) / std::log2(64.0 / 48.0);
    std::cerr << "[MMS-B vortex] rho e32=" << e32.rho << " e48=" << e48.rho
              << " e64=" << e64.rho
              << " rate(32->48)=" << r_3248
              << " rate(48->64)=" << r_4864 << "\n";
    // 32 is borderline-resolved (~3 cells/vortex scale) so 32->48 rate is
    // resolution-dominated (super-convergent). 48->64 should be in the
    // asymptotic regime; expect rate >= RK3 third-order or better.
    EXPECT_GT(r_3248, 2.5);
    EXPECT_GT(r_4864, 2.7);
}

// =================================================================
// MMS-C: steady viscous compressible NS solution with analytic source.
//
//   rho_M(x) = rho_0,                       u_M(x) = U_0 sin(x),
//   v_M = w_M = 0,                          p_M(x) = p_0.
//
// All quantities depend only on x. The exact analytic sources required for
// this to be a steady solution of the full compressible NS (mass + momentum
// + energy with constant mu Stokes-form viscous fluxes, kappa = mu cp / Pr):
//
//   S_rho = rho_0 U_0 cos x
//   S_mx  = rho_0 U_0^2 sin(2x) + (4/3) mu U_0 sin x
//   S_E   = (gamma p_0 / (gamma-1)) U_0 cos x
//         + (3 rho_0 U_0^3 / 8) (cos x - cos 3x)
//         - (4/3) mu U_0^2 cos(2x)
//
// (T = p_0 / (rho_0 R) is uniform so heat flux contributes zero -- one less
// term to derive.)
namespace mms_c {
constexpr Real U_0   = 0.1;
constexpr Real RHO_0 = 1.0;
constexpr Real P_0   = 1.0;
constexpr Real MU    = 0.01;
constexpr Real GAMMA = 1.4;
constexpr Real R_GAS = 1.0;

void fill_state(State& U, const Grid& g, const IdealGas& eos) {
    for (int k = -U.ng(); k < g.nz + U.ng(); ++k)
        for (int j = -U.ng(); j < g.ny + U.ng(); ++j)
            for (int i = -U.ng(); i < g.nx + U.ng(); ++i) {
                const Real x = g.xc(i);
                set_from_primitive(U, i, j, k, eos, RHO_0, U_0 * std::sin(x),
                                   0.0, 0.0, P_0);
            }
}

// Adds the analytic source to Rhs (NOT overwriting).
void add_source(State& Rhs, const Grid& g, Real /*t*/) {
    const Real gp_gm1 = GAMMA * P_0 / (GAMMA - 1.0);
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real x = g.xc(i);
                const Real cx = std::cos(x);
                const Real sx = std::sin(x);
                const Real c2x = std::cos(2.0 * x);
                const Real s2x = std::sin(2.0 * x);
                const Real c3x = std::cos(3.0 * x);

                const Real S_rho = RHO_0 * U_0 * cx;
                const Real S_mx  = RHO_0 * U_0 * U_0 * s2x
                                 + (4.0 / 3.0) * MU * U_0 * sx;
                const Real S_E   = gp_gm1 * U_0 * cx
                                 + (3.0 * RHO_0 * U_0 * U_0 * U_0 / 8.0)
                                   * (cx - c3x)
                                 - (4.0 / 3.0) * MU * U_0 * U_0 * c2x;

                Rhs[RHO ](i, j, k) += S_rho;
                Rhs[RHOU](i, j, k) += S_mx;
                Rhs[RHOE](i, j, k) += S_E;
                // v, w momentum sources are identically zero.
            }
}
}  // namespace mms_c

namespace {

Real run_mms_c(int N, Real t_end) {
    Grid g;
    g.nx = g.ny = g.nz = N;
    g.lx = g.ly = g.lz = 2.0 * M_PI;
    g.x0 = g.y0 = g.z0 = 0.0;

    GammaLaw gl; gl.gamma = mms_c::GAMMA; gl.R = mms_c::R_GAS;
    IdealGas eos{gl};
    State U(g.nx, g.ny, g.nz);
    mms_c::fill_state(U, g, eos);

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;
    apply_bcs(U, bc);

    ViscousParams vp;
    vp.mu = mms_c::MU;
    vp.prandtl = 0.71;
    vp.bulk_visc = 0.0;

    RK3 driver(g.nx, g.ny, g.nz, U.ng());

    SourceCallback src = mms_c::add_source;

    Real t = 0.0;
    while (t < t_end) {
        Real dt_hyp = max_dt_hyperbolic(U, g, eos, 0.4);
        Real dt_vis = max_dt_viscous(U, g, vp, 0.25);
        Real dt = std::min(dt_hyp, dt_vis);
        if (t + dt > t_end) dt = t_end - t;
        driver.step_with_source(U, g, bc, eos, vp, dt, t, src);
        t += dt;
    }

    // L2(rho - rho_M) over the interior. Since MMS-C is steady, the
    // analytic field equals the initial condition.
    Real l2 = 0.0;
    Real l2_mx = 0.0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real x = g.xc(i);
                const Real rho_an = mms_c::RHO_0;
                const Real mx_an  = mms_c::RHO_0 * mms_c::U_0 * std::sin(x);
                const Real e_rho = U[RHO ](i, j, k) - rho_an;
                const Real e_mx  = U[RHOU](i, j, k) - mx_an;
                l2 += e_rho * e_rho;
                l2_mx += e_mx * e_mx;
            }
    const Real N3 = static_cast<Real>(g.nx) * g.ny * g.nz;
    return std::sqrt((l2 + l2_mx) / N3);
}

}  // namespace

TEST(MMS3D, ViscousNSWithAnalyticSourceConvergesAtRK3Order) {
    // Steady manufactured solution: integrate for a few advective times so
    // that any drift from rho_M is dominated by the truncation error of the
    // discrete operator, not by transient.
    const Real t_end = 1.0;
    const Real e32 = run_mms_c(32, t_end);
    const Real e64 = run_mms_c(64, t_end);
    const Real rate = std::log2(e32 / e64);
    std::cerr << "[MMS-C viscous NS+source] e32=" << e32
              << " e64=" << e64 << " rate=" << rate << "\n";
    EXPECT_GT(rate, 2.7) << "expected RK3 third-order at fixed CFL";
    EXPECT_LT(e64, 1e-4);
}
