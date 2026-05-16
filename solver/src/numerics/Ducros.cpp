#include "numerics/Ducros.hpp"

#include <cmath>
#include <omp.h>

namespace blast {

void compute_ducros(const State& U, const Grid& g, Field3D& theta) {
    const int nx = g.nx, ny = g.ny, nz = g.nz;
    const Real inv_2dx = 1.0 / (2.0 * g.dx());
    const Real inv_2dy = (ny > 1) ? 1.0 / (2.0 * g.dy()) : 0.0;
    const Real inv_2dz = (nz > 1) ? 1.0 / (2.0 * g.dz()) : 0.0;

    const auto& rho = U[RHO];
    const auto& mx  = U[RHOU];
    const auto& my  = U[RHOV];
    const auto& mz  = U[RHOW];

    constexpr Real EPS_S = 1e-30;

    auto vel_at = [&](int i, int j, int k, Real& u, Real& v, Real& w) {
        const Real r = rho(i, j, k);
        u = mx(i, j, k) / r;
        v = my(i, j, k) / r;
        w = mz(i, j, k) / r;
    };

    // Compute sensor on interior + 1 ghost ring (so face-based dispatch can
    // read theta on both sides of every face up to the domain boundary).
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = -1; k < nz + 1; ++k)
        for (int j = -1; j < ny + 1; ++j)
            for (int i = -1; i < nx + 1; ++i) {
                Real u_xm, v_xm, w_xm, u_xp, v_xp, w_xp;
                vel_at(i - 1, j, k, u_xm, v_xm, w_xm);
                vel_at(i + 1, j, k, u_xp, v_xp, w_xp);

                Real du_dx = (u_xp - u_xm) * inv_2dx;
                Real dv_dx = (v_xp - v_xm) * inv_2dx;
                Real dw_dx = (w_xp - w_xm) * inv_2dx;

                Real du_dy = 0, dv_dy = 0, dw_dy = 0;
                if (ny > 1) {
                    Real u_ym, v_ym, w_ym, u_yp, v_yp, w_yp;
                    vel_at(i, j - 1, k, u_ym, v_ym, w_ym);
                    vel_at(i, j + 1, k, u_yp, v_yp, w_yp);
                    du_dy = (u_yp - u_ym) * inv_2dy;
                    dv_dy = (v_yp - v_ym) * inv_2dy;
                    dw_dy = (w_yp - w_ym) * inv_2dy;
                }
                Real du_dz = 0, dv_dz = 0, dw_dz = 0;
                if (nz > 1) {
                    Real u_zm, v_zm, w_zm, u_zp, v_zp, w_zp;
                    vel_at(i, j, k - 1, u_zm, v_zm, w_zm);
                    vel_at(i, j, k + 1, u_zp, v_zp, w_zp);
                    du_dz = (u_zp - u_zm) * inv_2dz;
                    dv_dz = (v_zp - v_zm) * inv_2dz;
                    dw_dz = (w_zp - w_zm) * inv_2dz;
                }

                const Real div = du_dx + dv_dy + dw_dz;

                // Vorticity components.
                const Real ox = dw_dy - dv_dz;
                const Real oy = du_dz - dw_dx;
                const Real oz = dv_dx - du_dy;
                const Real om2 = ox * ox + oy * oy + oz * oz;

                const Real div2 = div * div;
                theta(i, j, k) = div2 / (div2 + om2 + EPS_S);
            }
}

}  // namespace blast
