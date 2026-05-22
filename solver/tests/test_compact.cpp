#include <gtest/gtest.h>

#include "core/Types.hpp"
#include "numerics/CompactScheme.hpp"

#include <cmath>
#include <vector>

using blast::CompactPenta;
using blast::Real;

namespace {

// Smooth bump localized in the interior of [0,1] so the wall closure sees a
// (nearly) flat field -- isolates the 10th-order interior accuracy.
inline Real f_bump(Real x)  { Real z = (x - 0.5) / 0.12; return std::exp(-z * z); }
inline Real df_bump(Real x) { Real z = (x - 0.5) / 0.12; return std::exp(-z * z) * (-2.0 * z / 0.12); }

// L2 error of the conservative-difference derivative against the analytic one,
// measured over the interior cells only (away from the boundary closure).
Real deriv_err(int n) {
    const Real L = 1.0, dx = L / n;
    const int NG = blast::kCompactGhost;
    std::vector<Real> buf(n + 2 * NG);
    Real* node = buf.data() + NG;  // node[i], i in [-NG, n-1+NG]
    for (int i = -NG; i < n + NG; ++i) node[i] = f_bump((i + 0.5) * dx);

    std::vector<Real> faces(n + 1);
    CompactPenta cp(n);
    cp.reconstruct_wall(node, faces.data());

    const Real inv_dx = 1.0 / dx;
    Real l2 = 0.0;
    int cnt = 0;
    // Skip a fixed FRACTION near each wall: the 6th-order central6 closure
    // contaminates the global compact solve, but that influence decays with
    // distance, so at a fixed interior fraction it vanishes as n grows -- which
    // is what isolates the 10th-order interior rate.
    const int skip = n / 5;
    for (int i = skip; i < n - skip; ++i) {
        const Real est   = (faces[i + 1] - faces[i]) * inv_dx;  // d/dx at cell i
        const Real exact = df_bump((i + 0.5) * dx);
        const Real e = est - exact;
        l2 += e * e;
        ++cnt;
    }
    return std::sqrt(l2 / cnt);
}

}  // namespace

// Interior derivative converges at ~10th order.
TEST(Compact, TenthOrderInterior) {
    const Real e64  = deriv_err(64);
    const Real e128 = deriv_err(128);
    const Real rate = std::log2(e64 / e128);
    EXPECT_GT(rate, 9.0) << "e64=" << e64 << " e128=" << e128 << " rate=" << rate;
}

// Conservation: sum_i (Fhat_{i+1} - Fhat_i) telescopes to the net boundary flux
// exactly (to round-off), independent of the interior reconstruction.
TEST(Compact, TelescopesToBoundaryFlux) {
    const int n = 96;
    const Real dx = 1.0 / n;
    const int NG = blast::kCompactGhost;
    std::vector<Real> buf(n + 2 * NG);
    Real* node = buf.data() + NG;
    // A non-symmetric field so interior contributions are nontrivial.
    for (int i = -NG; i < n + NG; ++i) {
        const Real x = (i + 0.5) * dx;
        node[i] = std::sin(3.0 * x) + 0.4 * x * x;
    }
    std::vector<Real> faces(n + 1);
    CompactPenta cp(n);
    cp.reconstruct_wall(node, faces.data());

    Real sum = 0.0;
    for (int i = 0; i < n; ++i) sum += faces[i + 1] - faces[i];
    EXPECT_NEAR(sum, faces[n] - faces[0], 1e-12);
}

// Reconstruction is exact for a constant field (consistency: coeffs sum to 1).
TEST(Compact, ExactOnConstant) {
    const int n = 40;
    const int NG = blast::kCompactGhost;
    std::vector<Real> buf(n + 2 * NG, 3.5);
    Real* node = buf.data() + NG;
    std::vector<Real> faces(n + 1);
    CompactPenta cp(n);
    cp.reconstruct_wall(node, faces.data());
    for (int f = 0; f <= n; ++f) EXPECT_NEAR(faces[f], 3.5, 1e-12);
}
