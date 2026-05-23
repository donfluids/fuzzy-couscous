#pragma once

#include "core/Types.hpp"

#include <vector>

namespace blast {

// 10th-order conservative compact flux reconstruction (Lele pentadiagonal).
//
// Reconstructs cell-FACE fluxes Fhat_{i+1/2} from cell-CENTER fluxes F_i such
// that the conservative difference (Fhat_{i+1/2} - Fhat_{i-1/2}) / dx is the
// 10th-order compact first derivative of F at node i. Because each face carries
// exactly one value (shared by both neighbouring cells), the conservative
// difference telescopes -> discrete conservation is preserved bit-for-bit, just
// as for the explicit central6 reconstruction it replaces.
//
// Pentadiagonal relation (faces index f, node = cell center):
//   beta Fhat_{f-2} + alpha Fhat_{f-1} + Fhat_f + alpha Fhat_{f+1} + beta Fhat_{f+2}
//     = A (F_{f-1}+F_f) + B (F_{f-2}+F_{f+1}) + C (F_{f-3}+F_{f+2})
//   alpha=1/2, beta=1/20, A=527/600, B=17/100, C=1/600   (derivative-matched).
//
// Slip-wall closure: the two near-wall faces at each end (f = 0,1 and n-1,n) are
// taken from the explicit 6th-order central6 reconstruction (which reads BC
// ghost cells); the interior faces are solved with the compact relation. The
// constant LHS matrix is LU-factored once per line length and reused across all
// lines and flux components in a sweep (O(n) per solve).

// Interior compact coefficients (public so the RHS hybrid / tests can reuse).
inline constexpr Real kCompactAlpha = 0.5;            // 1/2
inline constexpr Real kCompactBeta  = 0.05;           // 1/20
inline constexpr Real kCompactA     = 527.0 / 600.0;  // inner node pair
inline constexpr Real kCompactB     = 17.0 / 100.0;   // mid node pair
inline constexpr Real kCompactC     = 1.0 / 600.0;    // outer node pair

// Number of ghost cells the reconstruction reads beyond each end of a line.
inline constexpr int kCompactGhost = 3;

class CompactPenta {
public:
    // n = number of interior cells along the line (>= 8 for a clean interior).
    explicit CompactPenta(int n);

    int n() const { return n_; }

    // Reconstruct n+1 face fluxes from a contiguous line of node fluxes.
    //   node  : pointer to cell 0; valid indices [-kCompactGhost .. n-1+kCompactGhost].
    //   faces : output, length n+1 (face f sits at x = f*dx, f = 0..n).
    // Slip-wall boundary closure (central6 at the four near-wall faces).
    void reconstruct_wall(const Real* node, Real* faces) const;

private:
    int n_;
    // Banded LU (bandwidth 2) of the constant (n+1)x(n+1) face matrix.
    std::vector<Real> l1_, l2_;       // unit-lower sub-diagonals
    std::vector<Real> u0_, u1_, u2_;  // upper diagonal + two super-diagonals
    void factor_wall();
};

}  // namespace blast
