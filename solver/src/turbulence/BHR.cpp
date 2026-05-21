#include "turbulence/BHR.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

namespace blast {

namespace {

inline Real vel(const State& U, int comp, int i, int j, int k) {
    return U[RHOU + comp](i, j, k) / U[RHO](i, j, k);
}

// Pressure from conserved state (ideal gas).
inline Real pres(const State& U, const IdealGas& eos, int i, int j, int k) {
    const Real rho = U[RHO](i, j, k);
    const Real u = U[RHOU](i, j, k) / rho;
    const Real v = U[RHOV](i, j, k) / rho;
    const Real w = U[RHOW](i, j, k) / rho;
    const Real ke = 0.5 * rho * (u * u + v * v + w * w);
    return eos.pressure(rho, U[RHOE](i, j, k) - ke);
}

// Mirror-fill ng ghost layers on one face with given parity (+1 even / -1 odd).
void fill_face(Field3D& F, int dim, int side, BCType type, int parity,
               int nx, int ny, int nz, int ng) {
    const int n = (dim == 0 ? nx : dim == 1 ? ny : nz);
    auto at = [&](int a, int b, int c) -> Real& {
        if (dim == 0) return F(a, b, c);
        if (dim == 1) return F(b, a, c);
        return F(b, c, a);
    };
    const int n1 = (dim == 0 ? ny : nx);
    const int n2 = (dim == 0 ? nz : dim == 1 ? nz : ny);
    for (int g = 1; g <= ng; ++g) {
        const int ghost = (side < 0 ? -g : n - 1 + g);
        int interior;
        if (type == BCType::Periodic) interior = (side < 0 ? n - g : g - 1);
        else                          interior = (side < 0 ? g - 1 : n - g);  // mirror
        const Real sgn = (type == BCType::Periodic) ? 1.0
                       : (type == BCType::Outflow)  ? 1.0
                       : Real(parity);               // SlipWall mirror parity
        for (int b = -ng; b < n1 + ng; ++b)
            for (int c = -ng; c < n2 + ng; ++c)
                at(ghost, b, c) = sgn * at(interior, b, c);
    }
}

void fill_field_bcs(Field3D& F, const BCSet& bc, int px, int py, int pz,
                    int nx, int ny, int nz, int ng) {
    fill_face(F, 0, -1, bc.xlo, px, nx, ny, nz, ng);
    fill_face(F, 0, +1, bc.xhi, px, nx, ny, nz, ng);
    fill_face(F, 1, -1, bc.ylo, py, nx, ny, nz, ng);
    fill_face(F, 1, +1, bc.yhi, py, nx, ny, nz, ng);
    fill_face(F, 2, -1, bc.zlo, pz, nx, ny, nz, ng);
    fill_face(F, 2, +1, bc.zhi, pz, nx, ny, nz, ng);
}

}  // namespace

void bhr_apply_bcs(TurbState& T, const BCSet& bc) {
    const int nx = T.nx(), ny = T.ny(), nz = T.nz(), ng = T.ng();
    // Scalars k,eps,b: even (Neumann). a_i: odd on its own axis, even otherwise.
    fill_field_bcs(T[TurbState::TK], bc, +1, +1, +1, nx, ny, nz, ng);
    fill_field_bcs(T[TurbState::TE], bc, +1, +1, +1, nx, ny, nz, ng);
    fill_field_bcs(T[TurbState::TB], bc, +1, +1, +1, nx, ny, nz, ng);
    fill_field_bcs(T[TurbState::TAX], bc, -1, +1, +1, nx, ny, nz, ng);
    fill_field_bcs(T[TurbState::TAY], bc, +1, -1, +1, nx, ny, nz, ng);
    fill_field_bcs(T[TurbState::TAZ], bc, +1, +1, -1, nx, ny, nz, ng);
}

void bhr_init(TurbState& T, const State& U, const Grid& g,
              const IdealGas& eos, const BHRParams& bp) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz();
    const Real dx = g.dx(), dy = g.dy(), dz = g.dz();
    for (int v = 0; v < TurbState::NT; ++v) T[v].fill(0.0);
    // First find max |grad rho| (contact-layer indicator).
    Real gmax = 0.0;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const Real gx = (U[RHO](i+1,j,k) - U[RHO](i-1,j,k)) / (2*dx);
                const Real gy = (U[RHO](i,j+1,k) - U[RHO](i,j-1,k)) / (2*dy);
                const Real gz = (U[RHO](i,j,k+1) - U[RHO](i,j,k-1)) / (2*dz);
                gmax = std::max(gmax, std::sqrt(gx*gx + gy*gy + gz*gz));
            }
    gmax = std::max(gmax, 1e-30);
    const Real len0 = 0.025;  // characteristic seed length (= bhr_rans_1d.py)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const Real rho = U[RHO](i,j,k);
                const Real p = pres(U, eos, i, j, k);
                const Real e_int = p / ((eos.eos.gamma - 1.0) * rho);
                const Real gx = (U[RHO](i+1,j,k) - U[RHO](i-1,j,k)) / (2*dx);
                const Real gy = (U[RHO](i,j+1,k) - U[RHO](i,j-1,k)) / (2*dy);
                const Real gz = (U[RHO](i,j,k+1) - U[RHO](i,j,k-1)) / (2*dz);
                const Real gr = std::sqrt(gx*gx + gy*gy + gz*gz) / gmax;
                const Real kk = std::max(bp.seed_scale * e_int * gr, bp.k_floor);
                T[TurbState::TK](i,j,k) = kk;
                T[TurbState::TE](i,j,k) =
                    std::max(bp.C_mu * std::pow(kk, 1.5) / len0, bp.e_floor);
                if (bp.b_seed > 0.0)
                    T[TurbState::TB](i,j,k) = bp.b_seed * gr;
            }
}

Real bhr_tke_integral(const State& U, const TurbState& T, const Grid& g) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz();
    const Real dV = g.cell_volume();
    Real s = 0.0;
#pragma omp parallel for collapse(2) reduction(+:s) schedule(static)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                s += U[RHO](i,j,k) * T[TurbState::TK](i,j,k) * dV;
    return s;
}

Real bhr_max_nu_t(const State& U, const TurbState& T, const BHRParams& bp) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz();
    Real m = 0.0;
#pragma omp parallel for collapse(2) reduction(max:m) schedule(static)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const Real kk = std::max(T[TurbState::TK](i,j,k), bp.k_floor);
                Real ee = std::max(T[TurbState::TE](i,j,k),
                                   bp.C_mu * std::pow(kk,1.5) / bp.L_max);
                const Real nut = bp.C_mu * kk * kk / std::max(ee, bp.e_floor);
                m = std::max(m, nut);
            }
    return m;
}

void bhr_peaks(const TurbState& T, Real& kmax, Real& amax, Real& bmax) {
    const int nx = T.nx(), ny = T.ny(), nz = T.nz();
    kmax = amax = bmax = 0.0;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                kmax = std::max(kmax, T[TurbState::TK](i,j,k));
                bmax = std::max(bmax, T[TurbState::TB](i,j,k));
                const Real ax = T[TurbState::TAX](i,j,k);
                const Real ay = T[TurbState::TAY](i,j,k);
                const Real az = T[TurbState::TAZ](i,j,k);
                amax = std::max(amax, std::sqrt(ax*ax + ay*ay + az*az));
            }
}

void bhr_write_radial_profiles(const State& U, const TurbState& T, const Grid& g,
                               const std::string& path, Real t) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz();
    constexpr int NB = 64;
    const Real r_max = 0.5;
    std::vector<double> cnt(NB,0), sk(NB,0), sar(NB,0), sb(NB,0);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const Real x = g.xc(i), y = g.yc(j), z = g.zc(k);
                const Real r = std::sqrt(x*x + y*y + z*z);
                if (r > r_max) continue;
                int bidx = int(r / r_max * NB);
                if (bidx < 0 || bidx >= NB) continue;
                const Real ri = 1.0 / std::max(r, 1e-30);
                const Real ar = T[TurbState::TAX](i,j,k)*x*ri
                              + T[TurbState::TAY](i,j,k)*y*ri
                              + T[TurbState::TAZ](i,j,k)*z*ri;
                cnt[bidx] += 1.0;
                sk[bidx]  += T[TurbState::TK](i,j,k);
                sar[bidx] += ar;
                sb[bidx]  += T[TurbState::TB](i,j,k);
            }
    std::ofstream os(path);
    os << "# t=" << t << "\nr,k,a_r,b\n";
    for (int bidx = 0; bidx < NB; ++bidx) {
        const double c = std::max(cnt[bidx], 1.0);
        const double rmid = (bidx + 0.5) / NB * r_max;
        os << rmid << ',' << sk[bidx]/c << ',' << sar[bidx]/c << ',' << sb[bidx]/c << '\n';
    }
}

Real bhr_write_sensor_profiles(const State& U, const TurbState& T, const Grid& g,
                               const std::string& path, Real t) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz();
    constexpr int NB = 64;
    const Real r_max = 0.5;
    std::vector<double> cnt(NB,0), sf(NB,0), skr(NB,0), sks(NB,0);
    double fk_sum = 0.0; long ncells = 0;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                // 3x3x3 box test-filter of velocity
                Real ub=0, vb=0, wb=0;
                for (int dk=-1; dk<=1; ++dk)
                    for (int dj=-1; dj<=1; ++dj)
                        for (int di=-1; di<=1; ++di) {
                            ub += vel(U,0,i+di,j+dj,k+dk);
                            vb += vel(U,1,i+di,j+dj,k+dk);
                            wb += vel(U,2,i+di,j+dj,k+dk);
                        }
                ub/=27.0; vb/=27.0; wb/=27.0;
                const Real du = vel(U,0,i,j,k)-ub, dv = vel(U,1,i,j,k)-vb, dw = vel(U,2,i,j,k)-wb;
                const Real k_res = 0.5*(du*du + dv*dv + dw*dw);
                const Real k_sgs = std::max(T[TurbState::TK](i,j,k), 0.0);
                const Real fk = k_sgs / std::max(k_res + k_sgs, 1e-30);
                fk_sum += fk; ++ncells;
                const Real x = g.xc(i), y = g.yc(j), z = g.zc(k);
                const Real r = std::sqrt(x*x + y*y + z*z);
                if (r > r_max) continue;
                int bidx = int(r / r_max * NB);
                if (bidx < 0 || bidx >= NB) continue;
                cnt[bidx]+=1.0; sf[bidx]+=fk; skr[bidx]+=k_res; sks[bidx]+=k_sgs;
            }
    std::ofstream os(path);
    os << "# t=" << t << "\nr,f_k,k_res,k_sgs\n";
    for (int bidx = 0; bidx < NB; ++bidx) {
        const double c = std::max(cnt[bidx], 1.0);
        os << (bidx+0.5)/NB*r_max << ',' << sf[bidx]/c << ','
           << skr[bidx]/c << ',' << sks[bidx]/c << '\n';
    }
    return ncells ? fk_sum / ncells : 0.0;
}

void bhr_substep(State& U, TurbState& T, const Grid& g, const BCSet& bc,
                 const IdealGas& eos, const BHRParams& bp, Real dt) {
    if (!bp.enabled) return;
    const int nx = U.nx(), ny = U.ny(), nz = U.nz(), ng = U.ng();
    const Real dx = g.dx(), dy = g.dy(), dz = g.dz();

    static TurbState Tn;
    static Field3D mut;
    if (Tn.nx() != nx || Tn.ny() != ny || Tn.nz() != nz) {
        Tn.allocate(nx, ny, nz, ng);
        mut.resize(nx, ny, nz, ng);
    }

    bhr_apply_bcs(T, bc);

    // Pass 1: eddy viscosity mu_t over interior + 1-cell halo (for face coeffs).
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = -1; k <= nz; ++k)
        for (int j = -1; j <= ny; ++j)
            for (int i = -1; i <= nx; ++i) {
                const Real rho = U[RHO](i,j,k);
                Real kk = std::max(T[TurbState::TK](i,j,k), bp.k_floor);
                Real ee = T[TurbState::TE](i,j,k);
                ee = std::max(ee, bp.C_mu * std::pow(kk, 1.5) / bp.L_max);
                mut(i,j,k) = bp.C_mu * rho * kk * kk / std::max(ee, bp.e_floor);
            }

    // Pass 2: advection + diffusion + sources -> Tn.
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const Real rho = U[RHO](i,j,k);
                const Real u = vel(U,0,i,j,k), v = vel(U,1,i,j,k), w = vel(U,2,i,j,k);

                // mean gradients (central)
                const Real dux = (vel(U,0,i+1,j,k)-vel(U,0,i-1,j,k))/(2*dx);
                const Real duy = (vel(U,0,i,j+1,k)-vel(U,0,i,j-1,k))/(2*dy);
                const Real duz = (vel(U,0,i,j,k+1)-vel(U,0,i,j,k-1))/(2*dz);
                const Real dvx = (vel(U,1,i+1,j,k)-vel(U,1,i-1,j,k))/(2*dx);
                const Real dvy = (vel(U,1,i,j+1,k)-vel(U,1,i,j-1,k))/(2*dy);
                const Real dvz = (vel(U,1,i,j,k+1)-vel(U,1,i,j,k-1))/(2*dz);
                const Real dwx = (vel(U,2,i+1,j,k)-vel(U,2,i-1,j,k))/(2*dx);
                const Real dwy = (vel(U,2,i,j+1,k)-vel(U,2,i,j-1,k))/(2*dy);
                const Real dwz = (vel(U,2,i,j,k+1)-vel(U,2,i,j,k-1))/(2*dz);
                const Real dpx = (pres(U,eos,i+1,j,k)-pres(U,eos,i-1,j,k))/(2*dx);
                const Real dpy = (pres(U,eos,i,j+1,k)-pres(U,eos,i,j-1,k))/(2*dy);
                const Real dpz = (pres(U,eos,i,j,k+1)-pres(U,eos,i,j,k-1))/(2*dz);
                const Real drx = (U[RHO](i+1,j,k)-U[RHO](i-1,j,k))/(2*dx);
                const Real dry = (U[RHO](i,j+1,k)-U[RHO](i,j-1,k))/(2*dy);
                const Real drz = (U[RHO](i,j,k+1)-U[RHO](i,j,k-1))/(2*dz);

                Real kk = std::max(T[TurbState::TK](i,j,k), bp.k_floor);
                Real ee = T[TurbState::TE](i,j,k);
                ee = std::max(ee, bp.C_mu * std::pow(kk,1.5) / bp.L_max);
                const Real ax = T[TurbState::TAX](i,j,k);
                const Real ay = T[TurbState::TAY](i,j,k);
                const Real az = T[TurbState::TAZ](i,j,k);
                const Real bb = T[TurbState::TB](i,j,k);
                const Real mt = mut(i,j,k);
                const Real inv_k = 1.0 / kk;
                const Real div = dux + dvy + dwz;

                // strain & modeled Reynolds stress R_ij = 2 mu_t (S_ij - div/3 d_ij) - 2/3 rho k d_ij
                auto Sij = [&](Real a, Real bcomp){ return 0.5*(a+bcomp); };
                const Real Sxx=dux, Syy=dvy, Szz=dwz;
                const Real Sxy=Sij(duy,dvx), Sxz=Sij(duz,dwx), Syz=Sij(dvz,dwy);
                const Real two3 = 2.0/3.0;
                const Real Rxx = 2*mt*(Sxx-div/3.0) - two3*rho*kk;
                const Real Ryy = 2*mt*(Syy-div/3.0) - two3*rho*kk;
                const Real Rzz = 2*mt*(Szz-div/3.0) - two3*rho*kk;
                const Real Rxy = 2*mt*Sxy, Rxz = 2*mt*Sxz, Ryz = 2*mt*Syz;

                // shear production P = R_ij dU_i/dx_j (Boussinesq)
                Real P_sh = Rxx*dux + Ryy*dvy + Rzz*dwz
                          + Rxy*(duy+dvx) + Rxz*(duz+dwx) + Ryz*(dvz+dwy);
                P_sh = std::clamp(P_sh, -bp.prod_limit*rho*ee, bp.prod_limit*rho*ee);
                // variable-density production (dominant). Sign matches the 1D
                // model's internal convention (bhr_rans_1d.py:316 P_b = a dp/dr,
                // with a generated by +b dp/dx): a aligns with grad p, so the
                // self-consistent TKE source is +a_i dp/dx_i (positive feedback).
                const Real P_VD = (ax*dpx + ay*dpy + az*dpz);
                const Real prod = P_sh + P_VD;

                // upwind advection of a primitive phi
                auto adv = [&](const Field3D& F){
                    const Real fx = (u>0) ? (F(i,j,k)-F(i-1,j,k))/dx : (F(i+1,j,k)-F(i,j,k))/dx;
                    const Real fy = (v>0) ? (F(i,j,k)-F(i,j-1,k))/dy : (F(i,j+1,k)-F(i,j,k))/dy;
                    const Real fz = (w>0) ? (F(i,j,k)-F(i,j,k-1))/dz : (F(i,j,k+1)-F(i,j,k))/dz;
                    return u*fx + v*fy + w*fz;
                };
                // diffusion (1/rho) div[(mu+mu_t/sigma) grad phi]
                auto diff = [&](const Field3D& F, Real sigma){
                    auto D=[&](int ii,int jj,int kk2){ return bp.mu_phys + mut(ii,jj,kk2)/sigma; };
                    const Real Dxe=0.5*(D(i,j,k)+D(i+1,j,k)), Dxw=0.5*(D(i,j,k)+D(i-1,j,k));
                    const Real Dyn=0.5*(D(i,j,k)+D(i,j+1,k)), Dys=0.5*(D(i,j,k)+D(i,j-1,k));
                    const Real Dzt=0.5*(D(i,j,k)+D(i,j,k+1)), Dzb=0.5*(D(i,j,k)+D(i,j,k-1));
                    const Real lap =
                        (Dxe*(F(i+1,j,k)-F(i,j,k)) - Dxw*(F(i,j,k)-F(i-1,j,k)))/(dx*dx)
                      + (Dyn*(F(i,j+1,k)-F(i,j,k)) - Dys*(F(i,j,k)-F(i,j-1,k)))/(dy*dy)
                      + (Dzt*(F(i,j,k+1)-F(i,j,k)) - Dzb*(F(i,j,k)-F(i,j,k-1)))/(dz*dz);
                    return lap / rho;
                };

                // --- k: explicit prod/diff/adv, explicit eps sink (as 1D) ---
                Real k_new = kk + dt*(prod/rho - ee + diff(T[TurbState::TK], bp.sigma_k))
                                - dt*adv(T[TurbState::TK]);
                k_new = std::max(k_new, bp.k_floor);

                // --- eps: point-implicit destruction ---
                const Real eps_src = (ee*inv_k)*(bp.C_e1*prod)/rho
                                   + diff(T[TurbState::TE], bp.sigma_e);
                Real e_new = (ee + dt*eps_src - dt*adv(T[TurbState::TE]))
                           / (1.0 + dt*bp.C_e2*ee*inv_k);
                e_new = std::max(e_new, bp.e_floor);

                // --- a_i: prod = b dp_i - (R_ij/rho) drho_j - rho a_j dU_i/dx_j ; sink -C_a (eps/k) a ---
                const Real aRx = (Rxx*drx + Rxy*dry + Rxz*drz)/rho;
                const Real aRy = (Rxy*drx + Ryy*dry + Ryz*drz)/rho;
                const Real aRz = (Rxz*drx + Ryz*dry + Rzz*drz)/rho;
                const Real ash_x = rho*(ax*dux + ay*duy + az*duz);
                const Real ash_y = rho*(ax*dvx + ay*dvy + az*dvz);
                const Real ash_z = rho*(ax*dwx + ay*dwy + az*dwz);
                const Real denom_a = 1.0 + dt*bp.C_a*ee*inv_k;
                Real ax_new = (ax + dt*((bb*dpx - aRx - ash_x)/rho
                              + diff(T[TurbState::TAX], bp.sigma_a)) - dt*adv(T[TurbState::TAX]))/denom_a;
                Real ay_new = (ay + dt*((bb*dpy - aRy - ash_y)/rho
                              + diff(T[TurbState::TAY], bp.sigma_a)) - dt*adv(T[TurbState::TAY]))/denom_a;
                Real az_new = (az + dt*((bb*dpz - aRz - ash_z)/rho
                              + diff(T[TurbState::TAZ], bp.sigma_a)) - dt*adv(T[TurbState::TAZ]))/denom_a;

                // --- b: prod = -2 a_i drho_i (1+b)/rho ; sink -C_b (eps/k) b ---
                const Real b_src = -2.0*(ax*drx + ay*dry + az*drz)*(1.0+bb)/rho
                                 + diff(T[TurbState::TB], bp.sigma_b);
                Real b_new = (bb + dt*b_src - dt*adv(T[TurbState::TB]))
                           / (1.0 + dt*bp.C_b*ee*inv_k);

                // realizability
                if (!std::isfinite(k_new)) k_new = bp.k_floor;
                if (!std::isfinite(e_new)) e_new = bp.e_floor;
                if (!std::isfinite(b_new)) b_new = 0.0;
                k_new = std::clamp(k_new, bp.k_floor, bp.k_max);
                e_new = std::max(e_new, bp.e_floor);
                b_new = std::clamp(b_new, 0.0, bp.b_max);
                const Real abound = std::sqrt(2.0*k_new*std::max(b_new,0.0)) + 1e-12;
                if (!std::isfinite(ax_new)) ax_new = 0.0;
                if (!std::isfinite(ay_new)) ay_new = 0.0;
                if (!std::isfinite(az_new)) az_new = 0.0;
                ax_new = std::clamp(ax_new, -abound, abound);
                ay_new = std::clamp(ay_new, -abound, abound);
                az_new = std::clamp(az_new, -abound, abound);

                Tn[TurbState::TK](i,j,k)  = k_new;
                Tn[TurbState::TE](i,j,k)  = e_new;
                Tn[TurbState::TAX](i,j,k) = ax_new;
                Tn[TurbState::TAY](i,j,k) = ay_new;
                Tn[TurbState::TAZ](i,j,k) = az_new;
                Tn[TurbState::TB](i,j,k)  = b_new;
            }

    for (int v = 0; v < TurbState::NT; ++v) T[v].swap(Tn[v]);

    // Optional HYBRID two-way feedback: add the f_k-blended turbulent-viscosity
    // momentum stress + its work + the modeled dissipative heating to the mean.
    // f_k (LES/RANS blend) scales the modeled stress so RANS acts where the grid
    // under-resolves (f_k->1) and recedes where the resolved LES carries the
    // energy (f_k->0). Eddy-viscosity (Laplacian) form for stability; the
    // turbulent-diffusion CFL is guarded in the driver via bhr_max_nu_t.
    if (bp.feedback) {
        bhr_apply_bcs(T, bc);
        static Field3D mueff, md0, md1, md2;
        if (mueff.nx() != nx || mueff.ny() != ny || mueff.nz() != nz) {
            mueff.resize(nx, ny, nz, ng);
            md0.resize(nx, ny, nz, ng); md1.resize(nx, ny, nz, ng);
            md2.resize(nx, ny, nz, ng);
        }
        // mu_eff = f_k * mu_t over interior + 1-cell halo
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = -1; k <= nz; ++k)
            for (int j = -1; j <= ny; ++j)
                for (int i = -1; i <= nx; ++i) {
                    Real ub = 0, vb = 0, wb = 0;
                    for (int dk = -1; dk <= 1; ++dk)
                        for (int dj = -1; dj <= 1; ++dj)
                            for (int di = -1; di <= 1; ++di) {
                                ub += vel(U,0,i+di,j+dj,k+dk);
                                vb += vel(U,1,i+di,j+dj,k+dk);
                                wb += vel(U,2,i+di,j+dj,k+dk);
                            }
                    ub /= 27.0; vb /= 27.0; wb /= 27.0;
                    const Real du = vel(U,0,i,j,k)-ub, dv = vel(U,1,i,j,k)-vb, dw = vel(U,2,i,j,k)-wb;
                    const Real k_res = 0.5*(du*du + dv*dv + dw*dw);
                    const Real k_sgs = std::max(T[TurbState::TK](i,j,k), 0.0);
                    const Real fk = k_sgs / std::max(k_res + k_sgs, 1e-30);
                    mueff(i,j,k) = fk * mut(i,j,k);
                }
        // compute momentum diffusion d_j(mu_eff d_j u_c) into md (no in-place race)
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    auto lap = [&](int c){
                        auto uc=[&](int a,int b,int d){ return vel(U,c,a,b,d); };
                        const Real Dxe=0.5*(mueff(i,j,k)+mueff(i+1,j,k)), Dxw=0.5*(mueff(i,j,k)+mueff(i-1,j,k));
                        const Real Dyn=0.5*(mueff(i,j,k)+mueff(i,j+1,k)), Dys=0.5*(mueff(i,j,k)+mueff(i,j-1,k));
                        const Real Dzt=0.5*(mueff(i,j,k)+mueff(i,j,k+1)), Dzb=0.5*(mueff(i,j,k)+mueff(i,j,k-1));
                        return (Dxe*(uc(i+1,j,k)-uc(i,j,k)) - Dxw*(uc(i,j,k)-uc(i-1,j,k)))/(dx*dx)
                             + (Dyn*(uc(i,j+1,k)-uc(i,j,k)) - Dys*(uc(i,j,k)-uc(i,j-1,k)))/(dy*dy)
                             + (Dzt*(uc(i,j,k+1)-uc(i,j,k)) - Dzb*(uc(i,j,k)-uc(i,j,k-1)))/(dz*dz);
                    };
                    md0(i,j,k)=lap(0); md1(i,j,k)=lap(1); md2(i,j,k)=lap(2);
                }
        // apply to momentum + energy (stress work + modeled dissipative heating)
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    const Real rho = U[RHO](i,j,k);
                    const Real fk = (mut(i,j,k) > 0 ? mueff(i,j,k)/mut(i,j,k) : 0.0);
                    const Real eps = std::max(T[TurbState::TE](i,j,k), bp.e_floor);
                    const Real u=vel(U,0,i,j,k), v=vel(U,1,i,j,k), w=vel(U,2,i,j,k);
                    U[RHOU](i,j,k) += dt * md0(i,j,k);
                    U[RHOV](i,j,k) += dt * md1(i,j,k);
                    U[RHOW](i,j,k) += dt * md2(i,j,k);
                    U[RHOE](i,j,k) += dt * (u*md0(i,j,k) + v*md1(i,j,k) + w*md2(i,j,k)
                                            + fk * rho * eps);
                }
    }
}

}  // namespace blast
