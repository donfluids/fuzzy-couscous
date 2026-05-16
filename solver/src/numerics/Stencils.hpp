#pragma once

#include "core/Types.hpp"

namespace blast::stencil {

inline constexpr int RADIUS = 3;
static_assert(RADIUS <= NGHOST, "stencil radius exceeds ghost layer");

// 6th-order central first derivative. Caller guarantees that stride*[-3..3]
// stays within allocated memory (ghost cells must be populated).
inline Real ddx_6(const Real* __restrict__ f, Index stride, Real inv_dx) {
    constexpr Real a1 = 45.0 / 60.0;
    constexpr Real a2 =  9.0 / 60.0;
    constexpr Real a3 =  1.0 / 60.0;
    return (a1 * (f[ stride] - f[-stride])
          - a2 * (f[2*stride] - f[-2*stride])
          + a3 * (f[3*stride] - f[-3*stride])) * inv_dx;
}

// 6th-order explicit Lele low-pass filter (a0=11/16, a1=15/32, a2=-3/16,
// a3=1/32; coefficients sum to 1). Fits within RADIUS ghost cells.
inline Real filter_6(const Real* __restrict__ f, Index stride) {
    constexpr Real c0 = 11.0 / 16.0;
    constexpr Real c1 = 15.0 / 64.0;
    constexpr Real c2 = -3.0 / 32.0;
    constexpr Real c3 =  1.0 / 64.0;
    return c0 *  f[0]
         + c1 * (f[ stride] + f[-stride])
         + c2 * (f[2*stride] + f[-2*stride])
         + c3 * (f[3*stride] + f[-3*stride]);
}

}  // namespace blast::stencil
