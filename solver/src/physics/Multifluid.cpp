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

    const Real delta = (mp.tanh_thickness > 0 ? mp.tanh_thickness : 1.5 * g.dx());
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

void mf_pressure_minmax(const State& U, const Field3D& G, Real& pmin, Real& pmax) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz();
    pmin = 1e300; pmax = -1e300;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const Real rho = U[RHO](i,j,k);
                const Real ke = 0.5*(U[RHOU](i,j,k)*U[RHOU](i,j,k)
                                   + U[RHOV](i,j,k)*U[RHOV](i,j,k)
                                   + U[RHOW](i,j,k)*U[RHOW](i,j,k))/rho;
                const Real p = (U[RHOE](i,j,k) - ke) / G(i,j,k);
                pmin = std::min(pmin, p); pmax = std::max(pmax, p);
            }
}

}  // namespace blast
