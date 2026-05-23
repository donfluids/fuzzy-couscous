#include "physics/Multifluid.hpp"

#ifdef BLAST_MPI
#include "bc/BC.hpp"
#endif

#include <algorithm>
#include <cmath>

namespace blast {

namespace {
// Real Y_4^2 angular shape (unnormalized): sin^2(theta)(7cos^2 theta - 1)cos(2phi).
inline Real Y42(Real x, Real y, Real z) {
    const Real r = std::sqrt(x*x + y*y + z*z) + 1e-30;
    const Real ct = z / r;                 // cos theta
    const Real s2 = std::max(1.0 - ct*ct, 0.0);  // sin^2 theta
    const Real phi = std::atan2(y, x);
    return s2 * (7.0 * ct*ct - 1.0) * std::cos(2.0 * phi);
}
}  // namespace

void mf_init_blast(State& U, Field3D& G, const Grid& g, const MultifluidParams& mp) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz();

    // JWL products (real high explosive, e.g. TNT): the marker is a products
    // mass fraction phi in [0,1] (not G). Products region = tabulated CJ state
    // (rho_cj, p_cj) with the JWL EOS; ambient = (rho_a, p_a) ideal air. The
    // bubble starts at rest (cj_u_frac ignored here -- least-stiff choice). The
    // internal energy is taken from the EOS selected by the frozen phi>=switch
    // rule so it is consistent with the flux-loop pressure.
    if (mp.jwl_mode) {
        const Real delta = (mp.tanh_thickness > 0 ? mp.tanh_thickness : 3.0 * g.dx());  // resolved (~3 dx) default: a ~1 dx contact grid-seeds RM
        const Real gam_a = mp.gamma_air;
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    const Real x = g.xc(i), y = g.yc(j), z = g.zc(k);
                    const Real r = std::sqrt(x*x + y*y + z*z);
                    const Real r0e = mp.r0 * (1.0 + mp.Y42_amp * Y42(x, y, z));
                    const Real w = 0.5 * (1.0 + std::tanh((r0e - r) / delta)); // 1 inside
                    const Real phi = w;
                    const Real rho = mp.rho_a + (mp.rho_cj - mp.rho_a) * w;
                    const Real p   = mp.p_a   + (mp.p_cj   - mp.p_a)   * w;
                    const Real e_int = (phi >= mp.phi_switch)
                                           ? jwl_eint_from_p(mp.jwl, rho, p)
                                           : p / (gam_a - 1.0);
                    U[RHO](i,j,k)  = rho;
                    U[RHOU](i,j,k) = 0.0;
                    U[RHOV](i,j,k) = 0.0;
                    U[RHOW](i,j,k) = 0.0;
                    U[RHOE](i,j,k) = e_int;          // velocity = 0 -> ke = 0
                    G(i,j,k)       = phi;
                }
        return;
    }

    const Real Ga = 1.0 / (mp.gamma_air - 1.0);
    const Real Gp = 1.0 / (mp.gamma_p   - 1.0);
    const Real pa = mp.rho_a * mp.R * mp.T_a;

    // Products state: either an arbitrary (rho_p, T_p) blob, or the
    // Chapman-Jouguet state of a detonation in the explosive (rho_e, T_e) with
    // heat release q and products gamma_p (Williams 1985 ideal-gas CJ).
    Real rho_prod = mp.rho_p, p_prod = mp.rho_p * mp.R * mp.T_p, u_cj = 0.0;
    if (mp.q > 0.0) {
        const Real gp = mp.gamma_p;
        const Real rho0 = (mp.rho_e > 0.0 ? mp.rho_e : mp.rho_p);
        const Real p0 = rho0 * mp.R * mp.T_e;
        const Real c0 = std::sqrt(gp * mp.R * mp.T_e);
        const Real alpha = (gp + 1.0) * mp.q / (c0 * c0);
        const Real MD2 = 1.0 + alpha + std::sqrt(alpha * alpha + 2.0 * alpha);
        const Real D = c0 * std::sqrt(MD2);
        p_prod   = p0 * (1.0 + gp * MD2) / (gp + 1.0);          // p_CJ
        rho_prod = rho0 * (gp + 1.0) * MD2 / (gp * MD2 + 1.0);  // rho_CJ
        u_cj     = D * (1.0 - rho0 / rho_prod);                 // outward particle vel
    }

    const Real delta = (mp.tanh_thickness > 0 ? mp.tanh_thickness : 3.0 * g.dx());  // resolved (~3 dx) default: a ~1 dx contact grid-seeds RM
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const Real x = g.xc(i), y = g.yc(j), z = g.zc(k);
                const Real r = std::sqrt(x*x + y*y + z*z);
                const Real r0e = mp.r0 * (1.0 + mp.Y42_amp * Y42(x, y, z));
                const Real w = 0.5 * (1.0 + std::tanh((r0e - r) / delta));  // 1 inside
                const Real rho = mp.rho_a + (rho_prod - mp.rho_a) * w;
                const Real p   = pa + (p_prod - pa) * w;
                const Real Gl  = Ga + (Gp - Ga) * w;        // volume-fraction blend of 1/(g-1)
                const Real inv_r = (r > 1e-12) ? 1.0 / r : 0.0;
                const Real ur = w * u_cj * mp.cj_u_frac;    // outward CJ particle velocity (fraction)
                const Real um = ur * x * inv_r, vm = ur * y * inv_r, wm = ur * z * inv_r;
                const Real ke = 0.5 * rho * (um*um + vm*vm + wm*wm);
                U[RHO](i,j,k)  = rho;
                U[RHOU](i,j,k) = rho * um;
                U[RHOV](i,j,k) = rho * vm;
                U[RHOW](i,j,k) = rho * wm;
                U[RHOE](i,j,k) = p * Gl + ke;               // e_int = p*G; + kinetic
                G(i,j,k)       = Gl;
            }
}

void mf_fill_G_bcs(Field3D& G, const BCSet& bc) {
    const int nx = G.nx(), ny = G.ny(), nz = G.nz(), ng = G.ng();
    auto face = [&](int dim, int side, BCType type) {
        const int n = (dim == 0 ? nx : dim == 1 ? ny : nz);
        const int n1 = (dim == 0 ? ny : nx), n2 = (dim == 0 ? nz : dim == 1 ? nz : ny);
        auto at = [&](int a, int b, int c) -> Real& {
            if (dim == 0) return G(a, b, c);
            if (dim == 1) return G(b, a, c);
            return G(b, c, a);
        };
        for (int gh = 1; gh <= ng; ++gh) {
            const int ghost = (side < 0 ? -gh : n - 1 + gh);
            const int inter = (type == BCType::Periodic)
                            ? (side < 0 ? n - gh : gh - 1)
                            : (side < 0 ? gh - 1 : n - gh);   // mirror (zero-grad)
            for (int b = -ng; b < n1 + ng; ++b)
                for (int c = -ng; c < n2 + ng; ++c)
                    at(ghost, b, c) = at(inter, b, c);
        }
    };
    face(0,-1,bc.xlo); face(0,+1,bc.xhi);
    face(1,-1,bc.ylo); face(1,+1,bc.yhi);
    face(2,-1,bc.zlo); face(2,+1,bc.zhi);
}

namespace {
// Upwind advection of G by the resolved velocity into a scratch buffer, then
// swap. Assumes G ghosts are already valid (caller fills them). Reads U at the
// cell centre only; reads G at +/-1 neighbours (interior + one ghost layer).
void advect_G_upwind(Field3D& G, const State& U, const Grid& g, Real dt) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz(), ng = U.ng();
    const Real dx = g.dx(), dy = g.dy(), dz = g.dz();
    static Field3D Gn;
    if (Gn.nx() != nx || Gn.ny() != ny || Gn.nz() != nz) Gn.resize(nx, ny, nz, ng);
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const Real rho = U[RHO](i,j,k);
                const Real u = U[RHOU](i,j,k)/rho, v = U[RHOV](i,j,k)/rho, w = U[RHOW](i,j,k)/rho;
                const Real gx = (u > 0) ? (G(i,j,k)-G(i-1,j,k))/dx : (G(i+1,j,k)-G(i,j,k))/dx;
                const Real gy = (ny>1) ? ((v > 0) ? (G(i,j,k)-G(i,j-1,k))/dy : (G(i,j+1,k)-G(i,j,k))/dy) : 0.0;
                const Real gz = (nz>1) ? ((w > 0) ? (G(i,j,k)-G(i,j,k-1))/dz : (G(i,j,k+1)-G(i,j,k))/dz) : 0.0;
                Gn(i,j,k) = G(i,j,k) - dt * (u*gx + v*gy + w*gz);
            }
    G.swap(Gn);
}
}  // namespace

void mf_advect_G(Field3D& G, const State& U, const Grid& g, const BCSet& bc, Real dt) {
    mf_fill_G_bcs(G, bc);         // valid ghosts for the upwind gradient
    advect_G_upwind(G, U, g, dt);
    mf_fill_G_bcs(G, bc);         // valid ghosts for the next RHS (reads gamma from G)
}

#ifdef BLAST_MPI
void mf_advect_G(Field3D& G, const State& U, const Grid& g, const BCSet& bc,
                 Real dt, const Domain& d, Halo& halo) {
    halo.exchange(G); apply_bcs(G, bc, d);   // interior-neighbour + physical ghosts
    advect_G_upwind(G, U, g, dt);
    halo.exchange(G); apply_bcs(G, bc, d);   // refresh ghosts for the next RHS
}
#endif

void mf_pressure_minmax(const State& U, const Field3D& G, Real& pmin, Real& pmax,
                        const MixtureEOS* mix) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz();
    pmin = 1e300; pmax = -1e300;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const Real rho = U[RHO](i,j,k);
                const Real ke = 0.5*(U[RHOU](i,j,k)*U[RHOU](i,j,k)
                                   + U[RHOV](i,j,k)*U[RHOV](i,j,k)
                                   + U[RHOW](i,j,k)*U[RHOW](i,j,k))/rho;
                const Real e_int = U[RHOE](i,j,k) - ke;
                Real p, c;
                if (mix) mix->p_c(G(i,j,k), rho, e_int, p, c);
                else     p = e_int / G(i,j,k);
                pmin = std::min(pmin, p); pmax = std::max(pmax, p);
            }
}

void mf_init_5eq(State& U, FiveEqAux& aux, const Grid& g,
                 const MultifluidParams& mp) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz();
    const MixtureEOS mix = mp.mixture();
    const Real delta = (mp.tanh_thickness > 0 ? mp.tanh_thickness : 3.0 * g.dx());
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const Real x = g.xc(i), y = g.yc(j), z = g.zc(k);
                const Real r = std::sqrt(x*x + y*y + z*z);
                const Real r0e = mp.r0 * (1.0 + mp.Y42_amp * Y42(x, y, z));
                const Real w = 0.5 * (1.0 + std::tanh((r0e - r) / delta));  // 1 inside
                Real a1 = mp.a1_out + (mp.a1_in - mp.a1_out) * w;
                a1 = std::min(std::max(a1, mix.a_floor), 1.0 - mix.a_floor);
                const Real Z1 = a1 * mp.rho1;
                const Real Z2 = (1.0 - a1) * mp.rho2;
                const Real rho = Z1 + Z2;
                const Real p = mp.p_out + (mp.p_in - mp.p_out) * w;
                const Real rhoe = mix.five_eq_rhoe_from_p(a1, Z1, Z2, p);
                const Real ke = 0.5 * rho * (mp.u0*mp.u0 + mp.v0*mp.v0 + mp.w0*mp.w0);
                U[RHO](i,j,k)  = rho;
                U[RHOU](i,j,k) = rho * mp.u0;
                U[RHOV](i,j,k) = rho * mp.v0;
                U[RHOW](i,j,k) = rho * mp.w0;
                U[RHOE](i,j,k) = rhoe + ke;
                aux.Z1(i,j,k) = Z1;
                aux.Z2(i,j,k) = Z2;
                aux.a1(i,j,k) = a1;
            }
}

void mf_fill_aux_bcs(FiveEqAux& aux, const BCSet& bc) {
    mf_fill_G_bcs(aux.Z1, bc);
    mf_fill_G_bcs(aux.Z2, bc);
    mf_fill_G_bcs(aux.a1, bc);
}

namespace {
// in-place out = a*x + b*y over the full padded field (interior + ghosts).
void field_axpby(Field3D& out, Real a, const Field3D& x, Real b, const Field3D& y) {
    const Index N = x.ldx() * (x.ny() + 2 * x.ng()) * (x.nz() + 2 * x.ng());
    const Real* __restrict__ xp = x.raw();
    const Real* __restrict__ yp = y.raw();
    Real* __restrict__ op = out.raw();
#pragma omp parallel for simd schedule(static)
    for (Index i = 0; i < N; ++i) op[i] = a * xp[i] + b * yp[i];
}
void field_axpbypcz(Field3D& out, Real a, const Field3D& x, Real b,
                    const Field3D& y, Real c, const Field3D& z) {
    const Index N = x.ldx() * (x.ny() + 2 * x.ng()) * (x.nz() + 2 * x.ng());
    const Real* __restrict__ xp = x.raw();
    const Real* __restrict__ yp = y.raw();
    const Real* __restrict__ zp = z.raw();
    Real* __restrict__ op = out.raw();
#pragma omp parallel for simd schedule(static)
    for (Index i = 0; i < N; ++i) op[i] = a * xp[i] + b * yp[i] + c * zp[i];
}
}  // namespace

void aux_axpby(FiveEqAux& out, Real a, const FiveEqAux& x, Real b,
               const FiveEqAux& y) {
    field_axpby(out.Z1, a, x.Z1, b, y.Z1);
    field_axpby(out.Z2, a, x.Z2, b, y.Z2);
    field_axpby(out.a1, a, x.a1, b, y.a1);
}

void aux_axpbypcz(FiveEqAux& out, Real a, const FiveEqAux& x, Real b,
                  const FiveEqAux& y, Real c, const FiveEqAux& z) {
    field_axpbypcz(out.Z1, a, x.Z1, b, y.Z1, c, z.Z1);
    field_axpbypcz(out.Z2, a, x.Z2, b, y.Z2, c, z.Z2);
    field_axpbypcz(out.a1, a, x.a1, b, y.a1, c, z.a1);
}

long enforce_5eq_bounds(State& U, FiveEqAux& aux, Real a_floor, Real z_floor) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz();
    long nclamp = 0;
#pragma omp parallel for collapse(2) schedule(static) reduction(+:nclamp)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                Real a1 = aux.a1(i,j,k);
                Real Z1 = aux.Z1(i,j,k);
                Real Z2 = aux.Z2(i,j,k);
                if (!(a1 >= a_floor))       { a1 = a_floor;       ++nclamp; }
                if (!(a1 <= 1.0 - a_floor)) { a1 = 1.0 - a_floor; ++nclamp; }
                if (!(Z1 >= z_floor))       { Z1 = z_floor;       ++nclamp; }
                if (!(Z2 >= z_floor))       { Z2 = z_floor;       ++nclamp; }
                aux.a1(i,j,k) = a1;
                aux.Z1(i,j,k) = Z1;
                aux.Z2(i,j,k) = Z2;
                U[RHO](i,j,k) = Z1 + Z2;   // keep mixture mass consistent
            }
    return nclamp;
}

GStats mf_g_stats(const Field3D& G) {
    const int nx = G.nx(), ny = G.ny(), nz = G.nz();
    Real gmin = 1e300, gmax = -1e300, sum = 0.0, sum2 = 0.0;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const Real g = G(i, j, k);
                gmin = std::min(gmin, g); gmax = std::max(gmax, g);
                sum += g; sum2 += g * g;
            }
    const long long N = static_cast<long long>(nx) * ny * nz;
    const Real mean = sum / N;
    const Real var = std::max(sum2 / N - mean * mean, 0.0);
    return {gmin, gmax, mean, var};
}

#ifdef BLAST_MPI
GStats mf_g_stats(const Field3D& G, long long N_global, MPI_Comm comm) {
    const int nx = G.nx(), ny = G.ny(), nz = G.nz();
    Real gmin = 1e300, gmax = -1e300, sum = 0.0, sum2 = 0.0;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const Real g = G(i, j, k);
                gmin = std::min(gmin, g); gmax = std::max(gmax, g);
                sum += g; sum2 += g * g;
            }
    Real lo = gmin, hi = gmax, s[2] = {sum, sum2}, sg[2];
    MPI_Allreduce(&gmin, &lo, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&gmax, &hi, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(s, sg, 2, MPI_DOUBLE, MPI_SUM, comm);
    const Real mean = sg[0] / N_global;
    const Real var = std::max(sg[1] / N_global - mean * mean, 0.0);
    return {lo, hi, mean, var};
}
#endif

}  // namespace blast
