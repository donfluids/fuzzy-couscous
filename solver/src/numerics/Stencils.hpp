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

// 6th-order central second derivative. Coefficients on f[-3..+3]:
//   (2, -27, 270, -490, 270, -27, 2) / (180 dx^2)
// Used by hyperdissipation (composed twice to give nabla^4).
inline Real d2dx2_6(const Real* __restrict__ f, Index stride, Real inv_dx2) {
    constexpr Real a3 = 2.0   / 180.0;
    constexpr Real a2 = -27.0 / 180.0;
    constexpr Real a1 = 270.0 / 180.0;
    constexpr Real a0 = -490.0 / 180.0;
    return inv_dx2 * (
        a3 * (f[ 3 * stride] + f[-3 * stride])
      + a2 * (f[ 2 * stride] + f[-2 * stride])
      + a1 * (f[ 1 * stride] + f[-1 * stride])
      + a0 *  f[0]);
}

// 4th-order central second derivative. Coefficients on f[-2..+2]:
//   (-1, 16, -30, 16, -1) / (12 dx^2)
// Radius 2 so three composed Laplacians fit in NGHOST = 6 -- used only by
// the nabla^6 hyperdissipation path.
inline Real d2dx2_4(const Real* __restrict__ f, Index stride, Real inv_dx2) {
    constexpr Real a2 = -1.0  / 12.0;
    constexpr Real a1 = 16.0  / 12.0;
    constexpr Real a0 = -30.0 / 12.0;
    return inv_dx2 * (
        a2 * (f[ 2 * stride] + f[-2 * stride])
      + a1 * (f[ 1 * stride] + f[-1 * stride])
      + a0 *  f[0]);
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
