#include <gtest/gtest.h>

#include "bc/BC.hpp"
#include "core/Config.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "diagnostics/FFT.hpp"
#include "diagnostics/Spectra.hpp"
#include "diagnostics/Statistics.hpp"
#include "ic/Canonical.hpp"
#include "physics/EOS.hpp"

#include <algorithm>
#include <cmath>

using namespace blast;

namespace {

State make_rogallo_state(int N, Real urms, Real k_peak, int seed,
                        Grid& g_out) {
    g_out.nx = g_out.ny = g_out.nz = N;
    g_out.lx = g_out.ly = g_out.lz = 2.0 * M_PI;
    g_out.x0 = g_out.y0 = g_out.z0 = 0.0;
    IdealGas eos{GammaLaw{}};
    State U(N, N, N);
    ic_rogallo_3d(U, g_out, eos, urms, k_peak, 1.0, 1.0, seed);
    BCSet bc; bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic; bc.zlo = bc.zhi = BCType::Periodic;
    apply_bcs(U, bc);
    return U;
}

}  // namespace

// Rogallo IC must be divergence-free by construction (helical projection).
// After applying it on a periodic cube the Helmholtz decomposition should
// show K_dil / K_sol << 1.
TEST(Rogallo, DivergenceFreeByConstruction) {
    Grid g;
    State U = make_rogallo_state(32, /*urms=*/1.0, /*k_peak=*/4.0, /*seed=*/7, g);

    FFT3DPlan plan(g.nx, g.ny, g.nz);
    auto h = helmholtz_decompose(U, g, plan);

    EXPECT_GT(h.K_sol, 0.1);
    EXPECT_LT(h.K_dil, 1e-5 * h.K_sol)
        << "K_sol = " << h.K_sol << ", K_dil = " << h.K_dil;
}

// urms in physical space should match the requested target after the
// internal rescale. Verifies the closed-form normalization isn't needed.
TEST(Rogallo, UrmsMatchesTarget) {
    Grid g;
    const Real target_urms = 0.5;
    State U = make_rogallo_state(32, target_urms, /*k_peak=*/3.0, /*seed=*/11, g);

    IdealGas eos{GammaLaw{}};
    auto s = velocity_stats(U, eos);
    EXPECT_NEAR(s.u_rms, target_urms, 1e-3);
}

// The shell-averaged energy spectrum should peak near the requested k_peak.
// Tolerance: the discrete-k spectrum samples in integer shells so the peak
// can sit one bin off the analytic k_peak.
TEST(Rogallo, SpectrumPeaksNearKPeak) {
    Grid g;
    const Real k_peak = 4.0;
    State U = make_rogallo_state(64, /*urms=*/1.0, k_peak, /*seed=*/13, g);

    FFT3DPlan plan(g.nx, g.ny, g.nz);
    auto sp = velocity_spectrum(U, g, plan);

    // Locate the peak bin (skip k=0 mean and the lowest few bins).
    int peak_bin = 0;
    Real peak_E = 0;
    for (int b = 2; b < static_cast<int>(sp.E.size()); ++b) {
        if (sp.E[b] > peak_E) { peak_E = sp.E[b]; peak_bin = b; }
    }
    EXPECT_NEAR(static_cast<Real>(peak_bin), k_peak, 1.5)
        << "peak found at bin " << peak_bin;
    EXPECT_GT(peak_E, 0.0);
}

// Different seeds should produce statistically distinct realizations
// (different per-mode amplitudes) while reproducing the same target urms.
TEST(Rogallo, SeedReproducibilityAndIndependence) {
    Grid g1, g2, g3;
    State U1 = make_rogallo_state(32, 1.0, 4.0, 100, g1);
    State U2 = make_rogallo_state(32, 1.0, 4.0, 100, g2);   // same seed
    State U3 = make_rogallo_state(32, 1.0, 4.0, 200, g3);   // different seed

    // Same seed -> bit-identical
    Real max_diff_same = 0.0;
    for (int v = 0; v < NCONS; ++v)
        for (int k = 0; k < g1.nz; ++k)
            for (int j = 0; j < g1.ny; ++j)
                for (int i = 0; i < g1.nx; ++i)
                    max_diff_same = std::max(max_diff_same,
                        std::fabs(U1[v](i, j, k) - U2[v](i, j, k)));
    EXPECT_LT(max_diff_same, 1e-14);

    // Different seed -> noticeably different field
    Real max_diff_diff = 0.0;
    for (int k = 0; k < g1.nz; ++k)
        for (int j = 0; j < g1.ny; ++j)
            for (int i = 0; i < g1.nx; ++i)
                max_diff_diff = std::max(max_diff_diff,
                    std::fabs(U1[RHOU](i, j, k) - U3[RHOU](i, j, k)));
    EXPECT_GT(max_diff_diff, 0.1);
}
