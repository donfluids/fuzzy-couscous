#include <gtest/gtest.h>

#include "bc/BC.hpp"
#include "core/Field3D.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "ic/Canonical.hpp"
#include "numerics/Ducros.hpp"
#include "numerics/RK3.hpp"
#include "physics/EOS.hpp"

#include <cmath>
#include <iostream>

using namespace blast;

namespace {

// Companion to test_order_smooth.cpp (6th-order central branch). Here we drive
// a smooth right-going acoustic wave (M ≈ 0.2) that keeps the Ducros sensor
// θ above the 0.65 threshold across ≈ 80–95% of the domain (narrow strips at
// div(u)=0 stay on central6). Self-convergence (block-average restriction in x)
// cancels nonlinear physical-model error and measures only the numerical rate.
//
// What the rate reveals — the assembled hybrid scheme on a flow with sensor
// flickering converges at ~2nd order, NOT WENO5's intrinsic 5th order. The
// limit comes from the sensor architecture: div(u) is built with a 2nd-order
// centered difference (Ducros.cpp:138), and the binary switch at θ=0.65
// introduces an O(dx²) uncertainty in the WENO/central boundary that dominates
// the L2 error. This is the assembled-scheme effective order; a pure WENO5
// unit test (no sensor) would expose its formal 5th order. dt = K·dx² keeps
// the SSP-RK3 temporal error subdominant so the measured rate is spatial.
void inline_acoustic_ic(State& U, const Grid& g, Real A, const IdealGas& eos) {
    const Real rho0 = 1.0, p0 = 1.0;
    const Real c0 = std::sqrt(eos.eos.gamma * p0 / rho0);
    const Real k_phys = 2.0 * M_PI / g.lx;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real x = g.xc(i);
                const Real s = std::sin(k_phys * x);
                const Real rho = rho0 + A * s;
                const Real u   = (A * c0 / rho0) * s;
                const Real p   = p0  + A * c0 * c0 * s;
                set_from_primitive(U, i, j, k, eos, rho, u, 0.0, 0.0, p);
            }
}

void run_acoustic(int N, Real A, Real t_end, State& U_out, Real& weno_frac_out) {
    Grid g;
    g.nx = N; g.ny = 4; g.nz = 4;
    g.lx = 1.0; g.ly = g.lx / N * 4; g.lz = g.ly;
    g.x0 = 0.0; g.y0 = 0.0; g.z0 = 0.0;

    IdealGas eos{GammaLaw{}};
    U_out.allocate(g.nx, g.ny, g.nz);
    inline_acoustic_ic(U_out, g, A, eos);

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;

    RK3 driver(g.nx, g.ny, g.nz, U_out.ng());

    const Real dx = g.lx / N;
    const Real K = 2.0;
    const int nsteps = std::max(1, (int)std::lround(t_end / (K * dx * dx)));
    const Real dt = t_end / nsteps;
    for (int s = 0; s < nsteps; ++s) driver.step(U_out, g, bc, eos, dt);

    // Sensor coverage: fraction of interior cells with θ ≥ 0.65 at t_end. This
    // is the diagnostic that tells us whether the WENO5 branch was actually
    // exercised (vs. the central6 branch covering for it).
    Field3D theta(g.nx, g.ny, g.nz, U_out.ng(), "theta");
    compute_sensor(U_out, g, eos, theta);
    long active = 0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
                if (theta(i, j, k) >= 0.65) ++active;
    weno_frac_out = Real(active) / Real(g.nx * g.ny * g.nz);
}

// Block-average ρ_fine by 2 in x (y,z identical at 4×4 on both grids); return
// the L2 norm of (ρ_coarse − R(ρ_fine)) over the coarse interior.
Real restrict_l2_diff(const State& Uc, const State& Uf) {
    const int Nc = Uc.nx();
    EXPECT_EQ(Uf.nx(), 2 * Nc);
    EXPECT_EQ(Uc.ny(), Uf.ny());
    EXPECT_EQ(Uc.nz(), Uf.nz());
    Real l2 = 0.0;
    const Field3D& rc = Uc[RHO];
    const Field3D& rf = Uf[RHO];
    for (int k = 0; k < Uc.nz(); ++k)
        for (int j = 0; j < Uc.ny(); ++j)
            for (int i = 0; i < Nc; ++i) {
                const Real Rf = 0.5 * (rf(2*i, j, k) + rf(2*i + 1, j, k));
                const Real e  = rc(i, j, k) - Rf;
                l2 += e * e;
            }
    return std::sqrt(l2 / (Nc * Uc.ny() * Uc.nz()));
}

}  // namespace

// On this hybrid path the assembled scheme converges at ~2nd order (limited by
// the sensor's 2nd-order discretization of div(u) coupled with the binary
// scheme switch at θ=0.65), not at WENO5's intrinsic 5. The test asserts the
// observed rate and prints the WENO-active sensor fraction so a future fix
// raising the sensor or transition order would be visible immediately.
TEST(OrderWenoSmooth, HybridSchemeOrderOnSmoothFlow) {
    State U32, U64, U128, U256;
    Real f32 = 0, f64 = 0, f128 = 0, f256 = 0;
    run_acoustic(32,  0.2, 0.1, U32,  f32);
    run_acoustic(64,  0.2, 0.1, U64,  f64);
    run_acoustic(128, 0.2, 0.1, U128, f128);
    run_acoustic(256, 0.2, 0.1, U256, f256);

    const Real e_3264   = restrict_l2_diff(U32,  U64);
    const Real e_64128  = restrict_l2_diff(U64,  U128);
    const Real e_128256 = restrict_l2_diff(U128, U256);
    const Real r_a = std::log2(e_3264  / e_64128);
    const Real r_b = std::log2(e_64128 / e_128256);

    std::cerr << "[OrderWenoSmooth]"
              << " e(32->64)="  << e_3264
              << " e(64->128)=" << e_64128
              << " e(128->256)="<< e_128256
              << " rate_a="     << r_a
              << " rate_b="     << r_b
              << " weno_frac(32,64,128,256)="
              << f32 << "," << f64 << "," << f128 << "," << f256 << "\n";

    EXPECT_GT(r_b, 1.7) << "hybrid scheme should converge at no worse than ~2nd order";
    EXPECT_LT(r_b, 3.0) << "if rate jumps, sensor or transition order has improved — retune";
    EXPECT_GE(f256, 0.80) << "sensor must actually drive most cells into the WENO branch";
    EXPECT_LT(e_128256, 5e-5);
}
