#include "diagnostics/Statistics.hpp"

#include "numerics/Gradients.hpp"

#include <cmath>
#include <omp.h>

namespace blast {

VelocityStats velocity_stats(const State& U, const IdealGas& eos) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz();
    const long N = static_cast<long>(nx) * ny * nz;

    Real sum_u[3] = {0, 0, 0};
    Real sum_rho = 0, sum_p = 0, sum_T = 0, sum_c = 0;
    Real sum_ke = 0;

#pragma omp parallel for collapse(2) schedule(static) \
    reduction(+:sum_u[:3],sum_rho,sum_p,sum_T,sum_c,sum_ke)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const Real r  = U[RHO ](i, j, k);
                const Real ui = U[RHOU](i, j, k) / r;
                const Real vi = U[RHOV](i, j, k) / r;
                const Real wi = U[RHOW](i, j, k) / r;
                const Real ke = 0.5 * r * (ui*ui + vi*vi + wi*wi);
                const Real p  = eos.pressure(r, U[RHOE](i, j, k) - ke);
                const Real T  = eos.temperature(r, p);
                const Real c  = eos.sound_speed(r, p);
                sum_u[0] += ui;
                sum_u[1] += vi;
                sum_u[2] += wi;
                sum_rho  += r;
                sum_p    += p;
                sum_T    += T;
                sum_c    += c;
                sum_ke   += ke;
            }

    VelocityStats s{};
    const Real inv_N = 1.0 / N;
    for (int d = 0; d < 3; ++d) s.u_mean[d] = sum_u[d] * inv_N;
    s.rho_mean = sum_rho * inv_N;
    s.p_mean   = sum_p   * inv_N;
    s.T_mean   = sum_T   * inv_N;
    s.c_mean   = sum_c   * inv_N;
    s.ke_total = sum_ke  * inv_N;

    // Second pass: fluctuations after removing the mean.
    Real sum_up2 = 0, sum_tke = 0;
#pragma omp parallel for collapse(2) schedule(static) reduction(+:sum_up2,sum_tke)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const Real r  = U[RHO ](i, j, k);
                const Real du = U[RHOU](i, j, k) / r - s.u_mean[0];
                const Real dv = U[RHOV](i, j, k) / r - s.u_mean[1];
                const Real dw = U[RHOW](i, j, k) / r - s.u_mean[2];
                const Real q2 = du*du + dv*dv + dw*dw;
                sum_up2 += q2;
                sum_tke += 0.5 * r * q2;
            }
    s.u_rms = std::sqrt(sum_up2 * inv_N);
    s.tke   = sum_tke * inv_N;
    s.M_t   = s.u_rms / s.c_mean;
    return s;
}

DissipationBudget dissipation_budget(const State& U, const Grid& g,
                                     const IdealGas& eos,
                                     const ViscousParams& vp) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz();
    const long N = static_cast<long>(nx) * ny * nz;

    CellGradients G;
    G.allocate(U.nx(), U.ny(), U.nz(), U.ng());
    compute_cell_gradients(U, g, eos, G);

    Real sum_omega2 = 0, sum_div2 = 0;
    Real sum_eps_total_w = 0;  // weighted by 1/rho so total dissipation per mass

#pragma omp parallel for collapse(2) schedule(static) reduction(+:sum_omega2,sum_div2,sum_eps_total_w)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const Real ux = G.du[0][0](i, j, k), uy = G.du[0][1](i, j, k), uz = G.du[0][2](i, j, k);
                const Real vx = G.du[1][0](i, j, k), vy = G.du[1][1](i, j, k), vz = G.du[1][2](i, j, k);
                const Real wx = G.du[2][0](i, j, k), wy = G.du[2][1](i, j, k), wz = G.du[2][2](i, j, k);

                const Real div = ux + vy + wz;
                const Real ox  = wy - vz;
                const Real oy  = uz - wx;
                const Real oz  = vx - uy;
                const Real om2 = ox*ox + oy*oy + oz*oz;

                // Symmetric rate-of-strain S_ij and full dissipation
                //   tau_ij d u_i/d x_j = 2 mu S_ij S_ij - (2/3) mu div^2
                const Real s11 = ux, s22 = vy, s33 = wz;
                const Real s12 = 0.5 * (uy + vx);
                const Real s13 = 0.5 * (uz + wx);
                const Real s23 = 0.5 * (vz + wy);
                const Real s2 = s11*s11 + s22*s22 + s33*s33
                              + 2.0 * (s12*s12 + s13*s13 + s23*s23);
                const Real tau_ddu = 2.0 * vp.mu * s2 - (2.0 / 3.0) * vp.mu * div * div;

                const Real r = U[RHO](i, j, k);
                sum_eps_total_w += tau_ddu / r;
                sum_omega2 += om2;
                sum_div2   += div * div;
            }

    DissipationBudget b{};
    const Real inv_N = 1.0 / N;
    b.omega2_mean = sum_omega2 * inv_N;
    b.div2_mean   = sum_div2   * inv_N;
    b.eps_total   = sum_eps_total_w * inv_N;
    // For constant mu, the standard solenoidal/dilatational split of viscous
    // dissipation uses kinematic viscosity nu = mu / rho. We evaluate at the
    // domain-average density as a first approximation.
    Real sum_rho = 0;
#pragma omp parallel for collapse(2) schedule(static) reduction(+:sum_rho)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                sum_rho += U[RHO](i, j, k);
    const Real rho_mean = sum_rho * inv_N;
    const Real nu = vp.mu / rho_mean;
    b.eps_sol = nu * b.omega2_mean;
    b.eps_dil = (4.0 / 3.0) * nu * b.div2_mean;
    return b;
}

}  // namespace blast
