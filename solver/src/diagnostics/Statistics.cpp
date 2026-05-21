#include "diagnostics/Statistics.hpp"

#include "numerics/Gradients.hpp"

#include <cmath>
#include <omp.h>

namespace blast {

VelocityStats velocity_stats(const State& U, const IdealGas& eos) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz();
    const long N = static_cast<long>(nx) * ny * nz;

    Real sum_u[3] = {0, 0, 0};
    Real sum_mom[3] = {0, 0, 0};
    Real sum_rho = 0, sum_p = 0, sum_T = 0, sum_c = 0;
    Real sum_ke = 0, sum_E = 0, sum_eint = 0;

#pragma omp parallel for collapse(2) schedule(static) \
    reduction(+:sum_u[:3],sum_mom[:3],sum_rho,sum_p,sum_T,sum_c,sum_ke,sum_E,sum_eint)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const Real r   = U[RHO ](i, j, k);
                const Real ui  = U[RHOU](i, j, k) / r;
                const Real vi  = U[RHOV](i, j, k) / r;
                const Real wi  = U[RHOW](i, j, k) / r;
                const Real ke  = 0.5 * r * (ui*ui + vi*vi + wi*wi);
                const Real rE  = U[RHOE](i, j, k);
                const Real ein = rE - ke;          // internal energy density rho e_int
                const Real p   = eos.pressure(r, ein);
                const Real T   = eos.temperature(r, p);
                const Real c   = eos.sound_speed(r, p);
                sum_u[0] += ui;
                sum_u[1] += vi;
                sum_u[2] += wi;
                sum_mom[0] += U[RHOU](i, j, k);   // momentum density rho u_i
                sum_mom[1] += U[RHOV](i, j, k);
                sum_mom[2] += U[RHOW](i, j, k);
                sum_rho  += r;
                sum_p    += p;
                sum_T    += T;
                sum_c    += c;
                sum_ke   += ke;
                sum_E    += rE;     // total energy density rho E
                sum_eint += ein;    // internal energy density, from field variables
            }

    VelocityStats s{};
    const Real inv_N = 1.0 / N;
    for (int d = 0; d < 3; ++d) s.u_mean[d] = sum_u[d] * inv_N;
    for (int d = 0; d < 3; ++d) s.mom[d]    = sum_mom[d] * inv_N;
    s.rho_mean = sum_rho * inv_N;
    s.p_mean   = sum_p   * inv_N;
    s.T_mean   = sum_T   * inv_N;
    s.c_mean   = sum_c   * inv_N;
    s.ke_total = sum_ke  * inv_N;
    s.e_total  = sum_E   * inv_N;     // < rho E >       (conserved monitor)
    s.e_int    = sum_eint * inv_N;    // < rho e_int >   (computed from the field)

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

#ifdef BLAST_MPI
VelocityStats velocity_stats(const State& U, const IdealGas& eos,
                             long long N_global, MPI_Comm comm) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz();

    // u,v,w, rho,p,T,c, ke, rhoE, rho_eint, mom_x,mom_y,mom_z
    Real sum[13] = {0,0,0,0,0,0,0,0,0,0,0,0,0};
#pragma omp parallel for collapse(2) schedule(static) reduction(+:sum[:13])
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const Real r  = U[RHO ](i, j, k);
                const Real mxc = U[RHOU](i, j, k);
                const Real myc = U[RHOV](i, j, k);
                const Real mzc = U[RHOW](i, j, k);
                const Real ui = mxc / r;
                const Real vi = myc / r;
                const Real wi = mzc / r;
                const Real ke = 0.5 * r * (ui*ui + vi*vi + wi*wi);
                const Real rE = U[RHOE](i, j, k);
                const Real ein = rE - ke;        // internal energy density
                const Real p  = eos.pressure(r, ein);
                sum[0] += ui; sum[1] += vi; sum[2] += wi;
                sum[3] += r;  sum[4] += p;
                sum[5] += eos.temperature(r, p);
                sum[6] += eos.sound_speed(r, p);
                sum[7] += ke;
                sum[8] += rE;    // total energy density
                sum[9] += ein;   // internal energy density (from field variables)
                sum[10] += mxc; sum[11] += myc; sum[12] += mzc;  // momentum density
            }
    MPI_Allreduce(MPI_IN_PLACE, sum, 13, MPI_DOUBLE, MPI_SUM, comm);

    VelocityStats s{};
    const Real inv_N = 1.0 / static_cast<Real>(N_global);
    s.u_mean[0] = sum[0] * inv_N;
    s.u_mean[1] = sum[1] * inv_N;
    s.u_mean[2] = sum[2] * inv_N;
    s.mom[0]    = sum[10] * inv_N;
    s.mom[1]    = sum[11] * inv_N;
    s.mom[2]    = sum[12] * inv_N;
    s.rho_mean  = sum[3] * inv_N;
    s.p_mean    = sum[4] * inv_N;
    s.T_mean    = sum[5] * inv_N;
    s.c_mean    = sum[6] * inv_N;
    s.ke_total  = sum[7] * inv_N;
    s.e_total   = sum[8] * inv_N;
    s.e_int     = sum[9] * inv_N;   // computed from the field, not e_total - ke

    Real fluc[2] = {0, 0};
#pragma omp parallel for collapse(2) schedule(static) reduction(+:fluc[:2])
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const Real r  = U[RHO ](i, j, k);
                const Real du = U[RHOU](i, j, k) / r - s.u_mean[0];
                const Real dv = U[RHOV](i, j, k) / r - s.u_mean[1];
                const Real dw = U[RHOW](i, j, k) / r - s.u_mean[2];
                const Real q2 = du*du + dv*dv + dw*dw;
                fluc[0] += q2;
                fluc[1] += 0.5 * r * q2;
            }
    MPI_Allreduce(MPI_IN_PLACE, fluc, 2, MPI_DOUBLE, MPI_SUM, comm);
    s.u_rms = std::sqrt(fluc[0] * inv_N);
    s.tke   = fluc[1] * inv_N;
    s.M_t   = (s.c_mean > 0) ? s.u_rms / s.c_mean : 0.0;
    return s;
}

DissipationBudget dissipation_budget(const State& U, const Grid& g,
                                     const IdealGas& eos,
                                     const ViscousParams& vp,
                                     long long N_global, MPI_Comm comm) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz();

    CellGradients G;
    G.allocate(U.nx(), U.ny(), U.nz(), U.ng());
    compute_cell_gradients(U, g, eos, G);

    Real sums[4] = {0, 0, 0, 0};  // omega2, div2, eps_total_w, rho
#pragma omp parallel for collapse(2) schedule(static) reduction(+:sums[:4])
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const Real ux = G.du[0][0](i, j, k), uy = G.du[0][1](i, j, k), uz = G.du[0][2](i, j, k);
                const Real vx = G.du[1][0](i, j, k), vy = G.du[1][1](i, j, k), vz = G.du[1][2](i, j, k);
                const Real wx = G.du[2][0](i, j, k), wy = G.du[2][1](i, j, k), wz = G.du[2][2](i, j, k);
                const Real div = ux + vy + wz;
                const Real ox = wy - vz, oy = uz - wx, oz = vx - uy;
                const Real om2 = ox*ox + oy*oy + oz*oz;
                const Real s11 = ux, s22 = vy, s33 = wz;
                const Real s12 = 0.5 * (uy + vx);
                const Real s13 = 0.5 * (uz + wx);
                const Real s23 = 0.5 * (vz + wy);
                const Real s2 = s11*s11 + s22*s22 + s33*s33
                              + 2.0 * (s12*s12 + s13*s13 + s23*s23);
                const Real tau_ddu = 2.0 * vp.mu * s2 - (2.0 / 3.0) * vp.mu * div * div;
                const Real r = U[RHO](i, j, k);
                sums[0] += om2;
                sums[1] += div * div;
                sums[2] += tau_ddu / r;
                sums[3] += r;
            }
    MPI_Allreduce(MPI_IN_PLACE, sums, 4, MPI_DOUBLE, MPI_SUM, comm);

    DissipationBudget b{};
    const Real inv_N = 1.0 / static_cast<Real>(N_global);
    b.omega2_mean = sums[0] * inv_N;
    b.div2_mean   = sums[1] * inv_N;
    b.eps_total   = sums[2] * inv_N;
    const Real rho_mean = sums[3] * inv_N;
    const Real nu = vp.mu / rho_mean;
    b.eps_sol = nu * b.omega2_mean;
    b.eps_dil = (4.0 / 3.0) * nu * b.div2_mean;
    return b;
}
#endif

}  // namespace blast
