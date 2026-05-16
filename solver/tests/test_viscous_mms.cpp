#include <gtest/gtest.h>

#include "bc/BC.hpp"
#include "core/Config.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "ic/Canonical.hpp"
#include "numerics/RHS.hpp"
#include "physics/EOS.hpp"

#include <cmath>

using namespace blast;

namespace {

// Smooth manufactured velocity field on [0, 2 pi]^3 with constant rho and T.
//   u = sin(x) cos(y) cos(z)
//   v = -cos(x) sin(y) cos(z) / 2
//   w = -cos(x) cos(y) sin(z) / 2
// Divergence: du/dx + dv/dy + dw/dz
//   = cos(x)cos(y)cos(z) - cos(x)cos(y)cos(z)/2 - cos(x)cos(y)cos(z)/2 = 0
// So this is solenoidal. With constant mu, viscous momentum force = mu * Laplacian(u_i).
//   Laplacian(u) = -3 sin(x)cos(y)cos(z)
//   Laplacian(v) = (3/2) cos(x)sin(y)cos(z)
//   Laplacian(w) = (3/2) cos(x)cos(y)sin(z)
// Expected RHS contribution = mu * Laplacian for each velocity component.
Real run_viscous_mms(int N, int comp) {
    Grid g;
    g.nx = g.ny = g.nz = N;
    g.lx = g.ly = g.lz = 2.0 * M_PI;
    g.x0 = g.y0 = g.z0 = 0.0;

    IdealGas eos{GammaLaw{}};
    State U(g.nx, g.ny, g.nz);

    auto fill_uvw = [&](int i, int j, int k, Real& uu, Real& vv, Real& ww) {
        const Real x = g.xc(i), y = g.yc(j), z = g.zc(k);
        uu =        std::sin(x) * std::cos(y) * std::cos(z);
        vv = -0.5 * std::cos(x) * std::sin(y) * std::cos(z);
        ww = -0.5 * std::cos(x) * std::cos(y) * std::sin(z);
    };

    const Real rho0 = 1.0;
    const Real p0   = 1.0;
    for (int k = -U.ng(); k < g.nz + U.ng(); ++k)
        for (int j = -U.ng(); j < g.ny + U.ng(); ++j)
            for (int i = -U.ng(); i < g.nx + U.ng(); ++i) {
                Real uu, vv, ww;
                fill_uvw(i, j, k, uu, vv, ww);
                set_from_primitive(U, i, j, k, eos, rho0, uu, vv, ww, p0);
            }

    State Rhs(g.nx, g.ny, g.nz);
    for (int v = 0; v < NCONS; ++v) Rhs.fill(v, 0.0);

    ViscousParams vp;
    vp.mu = 0.1;
    vp.prandtl = 0.71;
    vp.bulk_visc = 0.0;
    add_rhs_viscous(U, g, eos, vp, Rhs);

    // Compare to analytic: viscous-only momentum eqn d(rho u)/dt = mu * Lap(u).
    // For constant rho, that's also d(rho u)/dt for each component.
    // The exact viscous momentum source is mu * Laplacian(u_comp).
    auto exact_lap = [&](int c, int i, int j, int k) {
        const Real x = g.xc(i), y = g.yc(j), z = g.zc(k);
        if (c == 0) return -3.0 * std::sin(x) * std::cos(y) * std::cos(z);
        if (c == 1) return  1.5 * std::cos(x) * std::sin(y) * std::cos(z);
        return                1.5 * std::cos(x) * std::cos(y) * std::sin(z);
    };

    Real l2 = 0.0;
    int  cnt = 0;
    const int slab = N / 4;       // skip near-boundary cells (periodic ICs but reduce edge sample noise)
    for (int k = slab; k < g.nz - slab; ++k)
        for (int j = slab; j < g.ny - slab; ++j)
            for (int i = slab; i < g.nx - slab; ++i) {
                const Real expected = vp.mu * exact_lap(comp, i, j, k);
                const Real got = Rhs[RHOU + comp](i, j, k);
                const Real e = got - expected;
                l2 += e * e;
                ++cnt;
            }
    return std::sqrt(l2 / cnt);
}

}  // namespace

TEST(ViscousMMS, SixthOrderOnSolenoidalSinusoid) {
    // Velocity field is exact for the 6-th order operator if N is large
    // enough that 2 pi / N k_max is sufficiently resolved.
    Real e_x_32 = run_viscous_mms(32, 0);
    Real e_x_64 = run_viscous_mms(64, 0);
    Real rate = std::log2(e_x_32 / e_x_64);
    EXPECT_GT(rate, 5.5) << "rate=" << rate << " e32=" << e_x_32 << " e64=" << e_x_64;

    Real e_y_32 = run_viscous_mms(32, 1);
    Real e_y_64 = run_viscous_mms(64, 1);
    rate = std::log2(e_y_32 / e_y_64);
    EXPECT_GT(rate, 5.5) << "rate=" << rate;

    Real e_z_32 = run_viscous_mms(32, 2);
    Real e_z_64 = run_viscous_mms(64, 2);
    rate = std::log2(e_z_32 / e_z_64);
    EXPECT_GT(rate, 5.5) << "rate=" << rate;
}
