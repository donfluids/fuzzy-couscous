#include <gtest/gtest.h>

#include "core/Config.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "diagnostics/FFT.hpp"
#include "diagnostics/Spectra.hpp"
#include "ic/Canonical.hpp"
#include "physics/EOS.hpp"

#include <cmath>

using namespace blast;

namespace {

State make_state_on_box(int N, Real L, Real rho0) {
    State U(N, N, N);
    for (int v = 0; v < NCONS; ++v) U.fill(v, 0.0);
    (void)rho0;
    return U;
}

void set_uvw(State& U, const Grid& g, const IdealGas& eos, Real rho0,
             Real (*ufn)(Real, Real, Real),
             Real (*vfn)(Real, Real, Real),
             Real (*wfn)(Real, Real, Real)) {
    const Real p0 = 1.0;
    for (int k = -U.ng(); k < g.nz + U.ng(); ++k)
        for (int j = -U.ng(); j < g.ny + U.ng(); ++j)
            for (int i = -U.ng(); i < g.nx + U.ng(); ++i) {
                const Real x = g.xc(i), y = g.yc(j), z = g.zc(k);
                set_from_primitive(U, i, j, k, eos, rho0,
                                   ufn(x, y, z), vfn(x, y, z), wfn(x, y, z), p0);
            }
}

Real u_single_mode(Real x, Real y, Real z) { (void)y; (void)z; return std::cos(x); }
Real zero_fn(Real, Real, Real) { return 0.0; }

// Taylor-Green-like solenoidal field on [0, 2pi]^3:
//   u =  sin x cos y cos z
//   v = -cos x sin y cos z / 2
//   w = -cos x cos y sin z / 2
// div u = 0 by construction.
Real u_tg(Real x, Real y, Real z) { return  std::sin(x) * std::cos(y) * std::cos(z); }
Real v_tg(Real x, Real y, Real z) { return -0.5 * std::cos(x) * std::sin(y) * std::cos(z); }
Real w_tg(Real x, Real y, Real z) { return -0.5 * std::cos(x) * std::cos(y) * std::sin(z); }

// Compressive (radial-ish) field that has nonzero divergence.
//   u = sin x, v = sin y, w = sin z
// div u = cos x + cos y + cos z (nonzero). vorticity = 0.
Real u_compress(Real x, Real, Real) { return std::sin(x); }
Real v_compress(Real, Real y, Real) { return std::sin(y); }
Real w_compress(Real, Real, Real z) { return std::sin(z); }

}  // namespace

TEST(Spectrum, SingleModeIsolatedInExpectedBin) {
    const int N = 32;
    Grid g; g.nx = g.ny = g.nz = N;
    g.lx = g.ly = g.lz = 2.0 * M_PI;
    g.x0 = g.y0 = g.z0 = 0.0;

    IdealGas eos{GammaLaw{}};
    State U = make_state_on_box(N, g.lx, 1.0);
    set_uvw(U, g, eos, 1.0, u_single_mode, zero_fn, zero_fn);

    FFT3DPlan plan(N, N, N);
    auto sp = velocity_spectrum(U, g, plan);

    // u = cos(x): one Fourier mode at k=(1,0,0). Energy in bin |k|=1 only.
    EXPECT_GT(sp.E[1], 1e-3);
    EXPECT_LT(sp.E[0], 1e-12);
    for (int b = 2; b < static_cast<int>(sp.E.size()); ++b)
        EXPECT_LT(sp.E[b], 1e-12) << "leakage at bin " << b << ": " << sp.E[b];

    // Total energy: <(1/2) u^2> = (1/2) * <cos^2> = 1/4.
    Real total = 0;
    for (auto e : sp.E) total += e;
    EXPECT_NEAR(total, 0.25, 1e-10);
}

TEST(Helmholtz, SolenoidalFieldHasZeroDilatational) {
    const int N = 32;
    Grid g; g.nx = g.ny = g.nz = N;
    g.lx = g.ly = g.lz = 2.0 * M_PI;
    g.x0 = g.y0 = g.z0 = 0.0;

    IdealGas eos{GammaLaw{}};
    State U = make_state_on_box(N, g.lx, 1.0);
    set_uvw(U, g, eos, 1.0, u_tg, v_tg, w_tg);

    FFT3DPlan plan(N, N, N);
    auto h = helmholtz_decompose(U, g, plan);

    EXPECT_GT(h.K_sol, 0.01);
    EXPECT_LT(h.K_dil, 1e-20) << "K_dil = " << h.K_dil;
    EXPECT_NEAR(h.K_sol / (h.K_sol + h.K_dil), 1.0, 1e-12);
}

TEST(Helmholtz, IrrotationalFieldHasZeroSolenoidal) {
    const int N = 32;
    Grid g; g.nx = g.ny = g.nz = N;
    g.lx = g.ly = g.lz = 2.0 * M_PI;
    g.x0 = g.y0 = g.z0 = 0.0;

    IdealGas eos{GammaLaw{}};
    State U = make_state_on_box(N, g.lx, 1.0);
    set_uvw(U, g, eos, 1.0, u_compress, v_compress, w_compress);

    FFT3DPlan plan(N, N, N);
    auto h = helmholtz_decompose(U, g, plan);

    // u_i = sin(x_i) gives k = (1,0,0), (0,1,0), (0,0,1). Each mode is parallel
    // to its k vector, so each is purely dilatational. K_sol should be 0.
    EXPECT_GT(h.K_dil, 0.01);
    EXPECT_LT(h.K_sol, 1e-20) << "K_sol = " << h.K_sol;
}

TEST(Spectrum, ParsevalAgainstPhysicalSpace) {
    const int N = 32;
    Grid g; g.nx = g.ny = g.nz = N;
    g.lx = g.ly = g.lz = 2.0 * M_PI;
    g.x0 = g.y0 = g.z0 = 0.0;

    IdealGas eos{GammaLaw{}};
    State U = make_state_on_box(N, g.lx, 1.0);
    set_uvw(U, g, eos, 1.0, u_tg, v_tg, w_tg);

    FFT3DPlan plan(N, N, N);
    auto sp = velocity_spectrum(U, g, plan);

    Real spectral_total = 0;
    for (auto e : sp.E) spectral_total += e;

    // Physical-space mean of (1/2) u_i u_i.
    Real phys_total = 0;
    int  cnt = 0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real r = U[RHO](i, j, k);
                const Real u = U[RHOU](i, j, k) / r;
                const Real v = U[RHOV](i, j, k) / r;
                const Real w = U[RHOW](i, j, k) / r;
                phys_total += 0.5 * (u*u + v*v + w*w);
                ++cnt;
            }
    phys_total /= cnt;

    EXPECT_NEAR(spectral_total, phys_total, 1e-12);
}
