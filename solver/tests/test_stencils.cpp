#include <gtest/gtest.h>

#include "core/Field3D.hpp"
#include "numerics/Stencils.hpp"

#include <cmath>

using blast::Field3D;
using blast::Real;

// Fill an interior+ghost region with f(x) = sin(k x); verify the 6th-order
// derivative converges at the expected order.
TEST(Stencil, SixthOrderConvergesOnSin) {
    auto err_at = [](int n) {
        const Real L = 2.0 * M_PI;
        const Real dx = L / n;
        Field3D F(n, 1, 1, blast::NGHOST);
        for (int i = -F.ng(); i < n + F.ng(); ++i) {
            const Real x = (i + 0.5) * dx;   // wrap by periodicity, x can be outside
            F(i, 0, 0) = std::sin(x);
        }
        Real l2 = 0.0;
        for (int i = 0; i < n; ++i) {
            const Real x = (i + 0.5) * dx;
            const Real exact = std::cos(x);
            const Real* fp = &F(i, 0, 0);
            const Real est = blast::stencil::ddx_6(fp, 1, 1.0 / dx);
            const Real e = est - exact;
            l2 += e * e;
        }
        return std::sqrt(l2 / n);
    };

    const Real e64  = err_at(64);
    const Real e128 = err_at(128);
    const Real rate = std::log2(e64 / e128);
    EXPECT_GT(rate, 5.7);   // expect ~6.0; allow a little slack
}

// Filter should be the identity to machine precision on a uniform field.
TEST(Stencil, FilterIdentityOnConstant) {
    Field3D F(16, 1, 1);
    for (int i = -F.ng(); i < F.nx() + F.ng(); ++i) F(i, 0, 0) = 7.0;
    for (int i = 0; i < F.nx(); ++i) {
        const Real* fp = &F(i, 0, 0);
        EXPECT_NEAR(blast::stencil::filter_6(fp, 1), 7.0, 1e-14);
    }
}
