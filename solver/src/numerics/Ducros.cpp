#include "numerics/Ducros.hpp"

#include "physics/Multifluid.hpp"

#include <algorithm>
#include <cmath>
#include <omp.h>

namespace blast {

namespace {

// Map relative pressure jump |dp|/(p+p) onto [0,1] with a smooth ramp so that
// jumps >~ 10% saturate near 1, putting the cell solidly above the WENO
// activation threshold (0.65).
inline Real pressure_indicator(Real dp_over_psum) {
    constexpr Real k = 10.0;          // ramp slope
    return std::tanh(k * dp_over_psum);
}

inline Real pressure_at(const State& U, const IdealGas& eos, int i, int j, int k) {
    const Real r = U[RHO ](i, j, k);
    const Real u = U[RHOU](i, j, k) / r;
    const Real v = U[RHOV](i, j, k) / r;
    const Real w = U[RHOW](i, j, k) / r;
    const Real ke = 0.5 * r * (u * u + v * v + w * w);
    return eos.pressure(r, U[RHOE](i, j, k) - ke);
}

}  // namespace

void compute_sensor(const State& U, const Grid& g, const IdealGas& eos,
                    Field3D& theta, const Field3D* gfn, const MixtureEOS* mix,
                    const FiveEqAux* aux5) {
    // JWL products pressure is not (gamma-1) e_int, so a JWL multifluid must use
    // the local EOS for the pressure-jump / sound-speed terms. Two-gamma and
    // single-fluid keep the reference-gamma path (bit-identical to before).
    const bool use_mix = (mix && gfn && mix->mode == MixMode::JWL);
    // Five-equation: use the mixture pressure so a uniform-pressure contact (with
    // a jump in alpha/density, hence in e_int) is NOT flagged as a shock.
    const bool use_5eq = (mix && aux5 && mix->mode == MixMode::FiveEquation);
    const int nx = g.nx, ny = g.ny, nz = g.nz;
    const Real inv_2dx = 1.0 / (2.0 * g.dx());
    const Real inv_2dy = (ny > 1) ? 1.0 / (2.0 * g.dy()) : 0.0;
    const Real inv_2dz = (nz > 1) ? 1.0 / (2.0 * g.dz()) : 0.0;

    const auto& rho = U[RHO];
    const auto& mx  = U[RHOU];
    const auto& my  = U[RHOV];
    const auto& mz  = U[RHOW];

    // Floor on the Ducros denominator. A flat uniform velocity field has
    // div = omega = 0 to round-off; without a physically scaled floor the
    // ratio div^2/(div^2+omega^2) saturates near 1 at numerical noise.
    // Floor = eps_rel * (c_ref / dx)^2 puts the floor at the scale of a
    // physical gradient across one cell. Following Pirozzoli & Bernardini
    // (JFM 2014).
    constexpr Real EPS_REL = 1e-6;
    const Real dx_min = std::min({g.dx(),
                                  (ny > 1 ? g.dy() : 1e30),
                                  (nz > 1 ? g.dz() : 1e30)});

    auto vel_at = [&](int i, int j, int k, Real& u, Real& v, Real& w) {
        const Real r = rho(i, j, k);
        u = mx(i, j, k) / r;
        v = my(i, j, k) / r;
        w = mz(i, j, k) / r;
    };

    // Pressure at a cell, EOS-aware. Non-mix path is bit-identical to the
    // free pressure_at() helper above.
    auto p_at = [&](int i, int j, int k) -> Real {
        if (!use_mix && !use_5eq) return pressure_at(U, eos, i, j, k);
        const Real r = rho(i, j, k);
        const Real u = mx(i, j, k) / r, v = my(i, j, k) / r, w = mz(i, j, k) / r;
        const Real ke = 0.5 * r * (u * u + v * v + w * w);
        Real p, c;
        if (use_5eq)
            mix->p_c_5eq(aux5->a1(i,j,k), aux5->Z1(i,j,k), aux5->Z2(i,j,k),
                         r, U[RHOE](i, j, k) - ke, p, c);
        else
            mix->p_c((*gfn)(i, j, k), r, U[RHOE](i, j, k) - ke, p, c);
        return p;
    };

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
                const Real ox = dw_dy - dv_dz;
                const Real oy = du_dz - dw_dx;
                const Real oz = dv_dx - du_dy;
                const Real om2 = ox * ox + oy * oy + oz * oz;
                const Real div2 = div * div;
                const Real pc_local = p_at(i, j, k);
                Real c_local;
                if (use_5eq) {
                    const Real r = rho(i,j,k);
                    const Real uu = mx(i,j,k)/r, vv = my(i,j,k)/r, ww = mz(i,j,k)/r;
                    const Real ke = 0.5 * r * (uu*uu + vv*vv + ww*ww);
                    Real pp, cc;
                    mix->p_c_5eq(aux5->a1(i,j,k), aux5->Z1(i,j,k), aux5->Z2(i,j,k),
                                 r, U[RHOE](i,j,k) - ke, pp, cc);
                    c_local = cc;
                } else if (use_mix) {
                    c_local = mix->sound_speed((*gfn)(i, j, k), rho(i, j, k), pc_local);
                } else {
                    c_local = eos.sound_speed(rho(i, j, k), pc_local);
                }
                const Real eps_floor = EPS_REL * (c_local / dx_min) * (c_local / dx_min);
                const Real phi_v = div2 / (div2 + om2 + eps_floor);

                // Pressure-jump indicator: maximum normalized neighbor jump.
                const Real pc = pc_local;
                Real max_jump = 0.0;
                auto consider = [&](int ii, int jj, int kk) {
                    const Real pn = p_at(ii, jj, kk);
                    const Real j_ratio = std::fabs(pn - pc) / (pn + pc + 1e-30);
                    if (j_ratio > max_jump) max_jump = j_ratio;
                };
                consider(i - 1, j, k); consider(i + 1, j, k);
                if (ny > 1) { consider(i, j - 1, k); consider(i, j + 1, k); }
                if (nz > 1) { consider(i, j, k - 1); consider(i, j, k + 1); }
                const Real phi_p = pressure_indicator(max_jump);

                theta(i, j, k) = std::max(phi_v, phi_p);
            }
}

}  // namespace blast
