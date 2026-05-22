#include "numerics/CompactScheme.hpp"

#include "numerics/WENO5.hpp"

namespace blast {

CompactPenta::CompactPenta(int n) : n_(n) {
    factor_wall();
}

// Build and LU-factor the constant (M = n+1) x M pentadiagonal face matrix with
// identity rows at the four near-wall faces (0,1,n-1,n) and the compact relation
// [beta,alpha,1,alpha,beta] on the interior faces (2..n-2). Banded LU, no
// pivoting -- the compact symbol 1 + cos k + 0.1 cos 2k stays in [0.1, 2.1] so
// the matrix is well-conditioned (cond ~ 21).
void CompactPenta::factor_wall() {
    const int M = n_ + 1;
    // Assemble the five constant diagonals.
    // a2[i] = A(i,i-2), a1[i] = A(i,i-1), d[i] = A(i,i),
    // c1[i] = A(i,i+1), c2[i] = A(i,i+2).
    std::vector<Real> a2(M, 0.0), a1(M, 0.0), d(M, 1.0), c1(M, 0.0), c2(M, 0.0);
    for (int f = 2; f <= n_ - 2; ++f) {
        a2[f] = kCompactBeta;
        a1[f] = kCompactAlpha;
        d[f]  = 1.0;
        c1[f] = kCompactAlpha;
        c2[f] = kCompactBeta;
    }
    // Rows 0,1,n-1,n keep the identity defaults (d=1, off-diagonals 0).

    l1_.assign(M, 0.0);
    l2_.assign(M, 0.0);
    u0_.assign(M, 0.0);
    u1_.assign(M, 0.0);
    u2_.assign(M, 0.0);

    // Banded LU (Doolittle), bandwidth 2, no pivoting.
    u0_[0] = d[0];
    u1_[0] = c1[0];
    u2_[0] = c2[0];
    if (M > 1) {
        l1_[1] = a1[1] / u0_[0];
        u0_[1] = d[1] - l1_[1] * u1_[0];
        u1_[1] = c1[1] - l1_[1] * u2_[0];
        u2_[1] = c2[1];
    }
    for (int i = 2; i < M; ++i) {
        l2_[i] = a2[i] / u0_[i - 2];
        l1_[i] = (a1[i] - l2_[i] * u1_[i - 2]) / u0_[i - 1];
        u0_[i] = d[i] - l2_[i] * u2_[i - 2] - l1_[i] * u1_[i - 1];
        u1_[i] = (i + 1 < M) ? c1[i] - l1_[i] * u2_[i - 1] : 0.0;
        u2_[i] = c2[i];
    }
}

void CompactPenta::reconstruct_wall(const Real* node, Real* faces) const {
    const int M = n_ + 1;

    // Right-hand side per face.
    //   interior face f (2..n-2): compact node combination
    //   boundary faces 0,1,n-1,n : explicit central6 (reads BC ghosts)
    std::vector<Real> r(M);
    auto central6_face = [&](int f) {
        // Face f sits between cell f-1 and cell f; central6 pointer at cell f-1.
        return weno5::reconstruct_central6(node + (f - 1), 1);
    };
    r[0] = central6_face(0);
    r[1] = central6_face(1);
    for (int f = 2; f <= n_ - 2; ++f) {
        r[f] = kCompactA * (node[f - 1] + node[f])
             + kCompactB * (node[f - 2] + node[f + 1])
             + kCompactC * (node[f - 3] + node[f + 2]);
    }
    r[n_ - 1] = central6_face(n_ - 1);
    r[n_]     = central6_face(n_);

    // Forward solve L y = r (overwrite r with y).
    if (M > 1) r[1] -= l1_[1] * r[0];
    for (int i = 2; i < M; ++i) r[i] -= l1_[i] * r[i - 1] + l2_[i] * r[i - 2];

    // Back solve U x = y.
    faces[M - 1] = r[M - 1] / u0_[M - 1];
    if (M >= 2)
        faces[M - 2] = (r[M - 2] - u1_[M - 2] * faces[M - 1]) / u0_[M - 2];
    for (int i = M - 3; i >= 0; --i)
        faces[i] = (r[i] - u1_[i] * faces[i + 1] - u2_[i] * faces[i + 2]) / u0_[i];
}

}  // namespace blast
