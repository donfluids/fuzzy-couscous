#pragma once

#include "core/Types.hpp"

namespace blast::weno5 {

inline constexpr Real EPS = 1e-6;

// 5th-order WENO (Jiang & Shu 1996) reconstruction of the face value
// f_{i+1/2}^- using cells v[i-2], v[i-1], v[i], v[i+1], v[i+2].
// `v` points to v[i]; we read v[-2..+2].
inline Real reconstruct_left(const Real* __restrict__ v, Index stride) {
    const Real vm2 = v[-2 * stride];
    const Real vm1 = v[-1 * stride];
    const Real v0  = v[ 0];
    const Real vp1 = v[ 1 * stride];
    const Real vp2 = v[ 2 * stride];

    // Smoothness indicators.
    const Real d_m2_m1 = vm2 - 2.0 * vm1 + v0;
    const Real d_m2_0  = vm2 - 4.0 * vm1 + 3.0 * v0;
    const Real b0 = (13.0 / 12.0) * d_m2_m1 * d_m2_m1
                  + 0.25         * d_m2_0  * d_m2_0;

    const Real d_m1_0 = vm1 - 2.0 * v0 + vp1;
    const Real d_m1_1 = vm1 - vp1;
    const Real b1 = (13.0 / 12.0) * d_m1_0 * d_m1_0
                  + 0.25         * d_m1_1 * d_m1_1;

    const Real d_0_1  = v0 - 2.0 * vp1 + vp2;
    const Real d_0_p2 = 3.0 * v0 - 4.0 * vp1 + vp2;
    const Real b2 = (13.0 / 12.0) * d_0_1  * d_0_1
                  + 0.25         * d_0_p2 * d_0_p2;

    // Optimal linear weights for left-biased reconstruction.
    constexpr Real d0 = 1.0 / 10.0;
    constexpr Real d1 = 6.0 / 10.0;
    constexpr Real d2 = 3.0 / 10.0;

    const Real a0 = d0 / ((EPS + b0) * (EPS + b0));
    const Real a1 = d1 / ((EPS + b1) * (EPS + b1));
    const Real a2 = d2 / ((EPS + b2) * (EPS + b2));
    const Real inv_sum = 1.0 / (a0 + a1 + a2);

    const Real w0 = a0 * inv_sum;
    const Real w1 = a1 * inv_sum;
    const Real w2 = a2 * inv_sum;

    // Candidate reconstructions on each substencil.
    const Real p0 = ( 2.0 * vm2 - 7.0 * vm1 + 11.0 * v0 ) / 6.0;
    const Real p1 = (-1.0 * vm1 + 5.0 * v0  +  2.0 * vp1) / 6.0;
    const Real p2 = ( 2.0 * v0  + 5.0 * vp1 -  1.0 * vp2) / 6.0;

    return w0 * p0 + w1 * p1 + w2 * p2;
}

// Right-biased reconstruction f_{i+1/2}^+ from v[i-1..i+3].
// Implemented as left-biased reconstruction of the mirrored stencil.
inline Real reconstruct_right(const Real* __restrict__ v, Index stride) {
    const Real vm1 = v[-1 * stride];
    const Real v0  = v[ 0];
    const Real vp1 = v[ 1 * stride];
    const Real vp2 = v[ 2 * stride];
    const Real vp3 = v[ 3 * stride];

    const Real d2_p2_p3 = vp3 - 2.0 * vp2 + vp1;
    const Real d2_p3_p1 = vp3 - 4.0 * vp2 + 3.0 * vp1;
    const Real b0 = (13.0 / 12.0) * d2_p2_p3 * d2_p2_p3
                  + 0.25         * d2_p3_p1 * d2_p3_p1;

    const Real d1_p1_p2 = vp2 - 2.0 * vp1 + v0;
    const Real d1_p2_0  = vp2 - v0;
    const Real b1 = (13.0 / 12.0) * d1_p1_p2 * d1_p1_p2
                  + 0.25         * d1_p2_0  * d1_p2_0;

    const Real d0_0_p1 = vp1 - 2.0 * v0 + vm1;
    const Real d0_p1_m1 = 3.0 * vp1 - 4.0 * v0 + vm1;
    const Real b2 = (13.0 / 12.0) * d0_0_p1 * d0_0_p1
                  + 0.25         * d0_p1_m1 * d0_p1_m1;

    constexpr Real d0 = 1.0 / 10.0;
    constexpr Real d1 = 6.0 / 10.0;
    constexpr Real d2 = 3.0 / 10.0;

    const Real a0 = d0 / ((EPS + b0) * (EPS + b0));
    const Real a1 = d1 / ((EPS + b1) * (EPS + b1));
    const Real a2 = d2 / ((EPS + b2) * (EPS + b2));
    const Real inv_sum = 1.0 / (a0 + a1 + a2);

    const Real w0 = a0 * inv_sum;
    const Real w1 = a1 * inv_sum;
    const Real w2 = a2 * inv_sum;

    const Real p0 = ( 2.0 * vp3 - 7.0 * vp2 + 11.0 * vp1) / 6.0;
    const Real p1 = (-1.0 * vp2 + 5.0 * vp1 +  2.0 * v0 ) / 6.0;
    const Real p2 = ( 2.0 * vp1 + 5.0 * v0  -  1.0 * vm1) / 6.0;

    return w0 * p0 + w1 * p1 + w2 * p2;
}

// 6th-order central face reconstruction for the Shu finite-difference
// framework (Shu 1998, eq. 2.11): the cell values F_i are interpreted as
// cell averages of an implicit h(x), and h_{i+1/2} is recovered with the
// standard 6-point symmetric stencil. With this operator,
// (h_{i+1/2} - h_{i-1/2}) / dx is the high-order discrete approximation
// to dF/dx at cell i.
//
// Coefficients on f_{i-2}, f_{i-1}, f_i, f_{i+1}, f_{i+2}, f_{i+3} are
// 1/60, -8/60, 37/60, 37/60, -8/60, 1/60. NB these are NOT the point-value
// polynomial face-interpolation coefficients (150/256, ...).
inline Real reconstruct_central6(const Real* __restrict__ v, Index stride) {
    constexpr Real c0 = 37.0 / 60.0;   // inner pair (v_i, v_{i+1})
    constexpr Real c1 = -8.0 / 60.0;   // mid   pair (v_{i-1}, v_{i+2})
    constexpr Real c2 =  1.0 / 60.0;   // outer pair (v_{i-2}, v_{i+3})
    return c0 * (v[0]            + v[ 1 * stride])
         + c1 * (v[-1 * stride]  + v[ 2 * stride])
         + c2 * (v[-2 * stride]  + v[ 3 * stride]);
}

}  // namespace blast::weno5
