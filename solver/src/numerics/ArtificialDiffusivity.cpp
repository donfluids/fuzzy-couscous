#include "numerics/ArtificialDiffusivity.hpp"

#include "core/Types.hpp"
#include "numerics/Stencils.hpp"

#include <algorithm>
#include <cmath>

namespace blast {

namespace {

// Finite-safe raw 2nd difference (4th-order stencil, inv_dx2 = 1 so the result
// is ~ dx^2 d^2 f/dx^2). Unfilled edge/corner ghost cells (apply_bcs fills only
// face ghosts) hold garbage, so any non-finite stencil value is treated as 0.
// This keeps the LAD sensor finite everywhere; interior cells more than two
// cells from a boundary read only valid data and are unaffected.
inline Real d2_raw_safe(const Real* p, Index s) {
    auto z = [](Real x) { return std::isfinite(x) ? x : 0.0; };
    return (-1.0 * (z(p[2 * s]) + z(p[-2 * s]))
            + 16.0 * (z(p[s]) + z(p[-s]))
            - 30.0 * z(p[0])) / 12.0;
}

}  // namespace

Real compute_lad_fields(const State& U, const Grid& g, const IdealGas& eos,
                        const ViscousParams& vp, const CellGradients& Grad,
                        const Field3D& primT,
                        Field3D& theta_src, Field3D& strain_src,
                        Field3D& mu_art, Field3D& beta_art, Field3D& kappa_art) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz(), ng = U.ng();
    const auto& rho = U[RHO];

    const Real dx = g.dx();
    const Real dy = (ny > 1) ? g.dy() : dx;
    const Real dz = (nz > 1) ? g.dz() : dx;
    const Real hbar = std::cbrt(dx * dy * dz);   // representative length
    const Real h2   = hbar * hbar;
    const Real gamma = eos.eos.gamma;
    const Real Rgas  = eos.eos.R;
    const Real cv    = Rgas / (gamma - 1.0);
    const Real cp    = eos.eos.cp();

    // ---- Step 1: theta = div u and strain magnitude |S| on [-3, n+3) -----
    const int lo1   = -ng + stencil::RADIUS;
    const int hi1x  = nx + ng - stencil::RADIUS;
    const int hi1y  = ny + ng - stencil::RADIUS;
    const int hi1z  = nz + ng - stencil::RADIUS;
    theta_src.fill(0.0);
    strain_src.fill(0.0);
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = lo1; k < hi1z; ++k)
        for (int j = lo1; j < hi1y; ++j)
            for (int i = lo1; i < hi1x; ++i) {
                const Real d00 = Grad.du[0][0](i, j, k);
                const Real d11 = Grad.du[1][1](i, j, k);
                const Real d22 = Grad.du[2][2](i, j, k);
                theta_src(i, j, k) = d00 + d11 + d22;
                const Real sxy = 0.5 * (Grad.du[0][1](i, j, k) + Grad.du[1][0](i, j, k));
                const Real sxz = 0.5 * (Grad.du[0][2](i, j, k) + Grad.du[2][0](i, j, k));
                const Real syz = 0.5 * (Grad.du[1][2](i, j, k) + Grad.du[2][1](i, j, k));
                const Real s2  = d00 * d00 + d11 * d11 + d22 * d22
                               + 2.0 * (sxy * sxy + sxz * sxz + syz * syz);
                strain_src(i, j, k) = std::sqrt(2.0 * s2);
            }

    // ---- Step 2: LAD coefficient fields on [-1, n+1) ---------------------
    auto d2mag = [&](const Field3D& f, int i, int j, int k) -> Real {
        const Real* p = &f(i, j, k);
        Real m = d2_raw_safe(p, 1); m *= m;
        if (ny > 1) { const Real s = d2_raw_safe(p, f.ldx());  m += s * s; }
        if (nz > 1) { const Real s = d2_raw_safe(p, f.ldxy()); m += s * s; }
        return std::sqrt(m);
    };

    const int lo2 = -1;
    const int hx  = nx + 1;
    const int hy  = (ny > 1) ? ny + 1 : ny;
    const int hz  = (nz > 1) ? nz + 1 : nz;
    const int joff = (ny > 1) ? -1 : 0;
    const int koff = (nz > 1) ? -1 : 0;
    mu_art.fill(0.0);
    beta_art.fill(0.0);
    kappa_art.fill(0.0);

    Real nu_max = 0.0;
#pragma omp parallel for collapse(2) schedule(static) reduction(max : nu_max)
    for (int k = koff; k < hz; ++k)
        for (int j = joff; j < hy; ++j)
            for (int i = lo2; i < hx; ++i) {
                const Real r = rho(i, j, k);
                const Real T = primT(i, j, k);
                // Skip unfilled edge/corner ghost cells (garbage rho/T). Their
                // LAD is left at 0; the compact divergence never reads them for
                // interior updates, and this keeps nu_max finite.
                if (!std::isfinite(r) || r <= 0.0 || !std::isfinite(T) || T <= 0.0)
                    continue;

                const Real Mth = d2mag(theta_src, i, j, k);
                const Real MS  = d2mag(strain_src, i, j, k);
                const Real MT  = d2mag(primT, i, j, k);

                const Real th  = theta_src(i, j, k);
                const Real fsw = (std::isfinite(th) && th < 0.0) ? 1.0 : 0.0;
                const Real cs  = std::sqrt(gamma * Rgas * T);

                const Real mua = vp.abv_cmu    * r * MS * h2;
                const Real bta = vp.abv_cbeta  * r * fsw * Mth * h2;
                const Real kpa = vp.abv_ckappa * (r * cs / T) * (cv * MT) * hbar;

                mu_art(i, j, k)    = mua;
                beta_art(i, j, k)  = bta;
                kappa_art(i, j, k) = kpa;

                const Real inv_r  = 1.0 / r;
                const Real nu_mom = (mua + std::fabs(bta)) * inv_r;
                const Real nu_th  = kpa * inv_r / std::max(cp, 1e-30);
                const Real nu_loc = std::max(nu_mom, nu_th);
                if (nu_loc > nu_max) nu_max = nu_loc;
            }

    return nu_max;
}

}  // namespace blast
