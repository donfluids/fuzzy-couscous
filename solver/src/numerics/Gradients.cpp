#include "numerics/Gradients.hpp"

#include "numerics/Stencils.hpp"

#include <omp.h>

namespace blast {

void CellGradients::allocate(int nx, int ny, int nz, int ng) {
    for (int v = 0; v < 3; ++v)
        for (int d = 0; d < 3; ++d) du[v][d].resize(nx, ny, nz, ng);
    for (int d = 0; d < 3; ++d) dT[d].resize(nx, ny, nz, ng);
}

void compute_cell_gradients(const State& U, const Grid& g, const IdealGas& eos,
                            Field3D& u, Field3D& v, Field3D& w, Field3D& T,
                            CellGradients& G) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz(), ng = U.ng();

    // We can compute gradients at every cell whose 6-point stencil
    // (radius 3) lies within the populated region. U has ng ghosts, so
    // gradients are valid on cells where i in [-ng+3, n+ng-3].
    const int lo = -ng + stencil::RADIUS;
    const int hi_x = nx + ng - stencil::RADIUS;
    const int hi_y = ny + ng - stencil::RADIUS;
    const int hi_z = nz + ng - stencil::RADIUS;

    const Real inv_dx = 1.0 / g.dx();
    const Real inv_dy = 1.0 / g.dy();
    const Real inv_dz = 1.0 / g.dz();

    const auto& rho = U[RHO];
    const auto& mx  = U[RHOU];
    const auto& my  = U[RHOV];
    const auto& mz  = U[RHOW];
    const auto& E   = U[RHOE];

#pragma omp parallel for collapse(2) schedule(static)
    for (int k = -ng; k < nz + ng; ++k)
        for (int j = -ng; j < ny + ng; ++j)
#pragma omp simd
            for (int i = -ng; i < nx + ng; ++i) {
                const Real r  = rho(i, j, k);
                const Real ui = mx(i, j, k) / r;
                const Real vi = my(i, j, k) / r;
                const Real wi = mz(i, j, k) / r;
                const Real ke = 0.5 * r * (ui*ui + vi*vi + wi*wi);
                const Real p  = eos.pressure(r, E(i, j, k) - ke);
                u(i, j, k) = ui;
                v(i, j, k) = vi;
                w(i, j, k) = wi;
                T(i, j, k) = eos.temperature(r, p);
            }

    auto take_derivs_along = [&](int d, Real inv_dh) {
        const Index s = (d == 0 ? 1 : (d == 1 ? u.ldx() : u.ldxy()));
        Field3D& du_d = G.du[0][d];
        Field3D& dv_d = G.du[1][d];
        Field3D& dw_d = G.du[2][d];
        Field3D& dT_d = G.dT[d];

#pragma omp parallel for collapse(2) schedule(static)
        for (int k = lo; k < hi_z; ++k)
            for (int j = lo; j < hi_y; ++j)
#pragma omp simd
                for (int i = lo; i < hi_x; ++i) {
                    du_d(i,j,k) = stencil::ddx_6(&u(i,j,k), s, inv_dh);
                    dv_d(i,j,k) = stencil::ddx_6(&v(i,j,k), s, inv_dh);
                    dw_d(i,j,k) = stencil::ddx_6(&w(i,j,k), s, inv_dh);
                    dT_d(i,j,k) = stencil::ddx_6(&T(i,j,k), s, inv_dh);
                }
    };

    take_derivs_along(0, inv_dx);
    take_derivs_along(1, inv_dy);
    take_derivs_along(2, inv_dz);
}

void compute_cell_gradients(const State& U, const Grid& g, const IdealGas& eos,
                            CellGradients& G) {
    // Standalone wrapper for tests / ad-hoc callers. Allocates the primitive
    // scratch once on the stack and forwards to the scratch-aware overload.
    const int nx = U.nx(), ny = U.ny(), nz = U.nz(), ng = U.ng();
    Field3D u(nx, ny, nz, ng), v(nx, ny, nz, ng), w(nx, ny, nz, ng), T(nx, ny, nz, ng);
    compute_cell_gradients(U, g, eos, u, v, w, T, G);
}

}  // namespace blast
