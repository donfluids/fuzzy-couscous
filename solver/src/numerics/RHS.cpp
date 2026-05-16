#include "numerics/RHS.hpp"

#include "core/Field3D.hpp"
#include "numerics/Stencils.hpp"
#include "physics/EOS.hpp"
#include "physics/EulerFlux.hpp"

#include <algorithm>
#include <cmath>
#include <omp.h>

namespace blast {

namespace {

// Compute flux component v in direction d at all cells (including ghosts where
// possible) and store in `Fx`. The d-stencil derivative will then be taken.
// We compute on the full padded region [-ng, n+ng) so derivatives in the
// interior work even when ghost layer is BC-filled.
void compute_flux_dir(const State& U, int d, const IdealGas& eos, State& Flux) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz(), ng = U.ng();
    const auto& rho = U[RHO];
    const auto& mx  = U[RHOU];
    const auto& my  = U[RHOV];
    const auto& mz  = U[RHOW];
    const auto& E   = U[RHOE];
    Field3D& Fr = Flux[RHO];
    Field3D& Fu = Flux[RHOU];
    Field3D& Fv = Flux[RHOV];
    Field3D& Fw = Flux[RHOW];
    Field3D& FE = Flux[RHOE];

#pragma omp parallel for collapse(2) schedule(static)
    for (int k = -ng; k < nz + ng; ++k)
        for (int j = -ng; j < ny + ng; ++j) {
#pragma omp simd
            for (int i = -ng; i < nx + ng; ++i) {
                ConsCell C{rho(i,j,k), mx(i,j,k), my(i,j,k), mz(i,j,k), E(i,j,k)};
                const Real inv_rho = 1.0 / C.rho;
                const Real u = C.mx * inv_rho;
                const Real v = C.my * inv_rho;
                const Real w = C.mz * inv_rho;
                const Real ke = 0.5 * C.rho * (u*u + v*v + w*w);
                const Real p  = eos.pressure(C.rho, C.rhoE - ke);
                FluxVec F = euler_flux(C, p, d);
                Fr(i,j,k) = F.f[RHO];
                Fu(i,j,k) = F.f[RHOU];
                Fv(i,j,k) = F.f[RHOV];
                Fw(i,j,k) = F.f[RHOW];
                FE(i,j,k) = F.f[RHOE];
            }
        }
}

// Stride in linear memory for direction d.
Index stride_for(int d, const Field3D& f) {
    if (d == 0) return 1;
    if (d == 1) return f.ldx();
    return f.ldxy();
}

void add_flux_divergence(const State& Flux, int d, Real inv_dh, State& Rhs) {
    const int nx = Rhs.nx(), ny = Rhs.ny(), nz = Rhs.nz();
    for (int v = 0; v < NCONS; ++v) {
        const Field3D& F = Flux[v];
        Field3D&       R = Rhs[v];
        const Index s = stride_for(d, F);
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j) {
#pragma omp simd
                for (int i = 0; i < nx; ++i) {
                    const Real* fp = &F(i, j, k);
                    R(i, j, k) -= stencil::ddx_6(fp, s, inv_dh);
                }
            }
    }
}

}  // namespace

void compute_rhs_inviscid(const State& U, const Grid& g, const IdealGas& eos,
                          State& Rhs) {
    for (int v = 0; v < NCONS; ++v) Rhs.fill(v, 0.0);

    State Flux(U.nx(), U.ny(), U.nz(), U.ng());

    compute_flux_dir(U, 0, eos, Flux);
    add_flux_divergence(Flux, 0, 1.0 / g.dx(), Rhs);

    if (U.ny() > 1) {
        compute_flux_dir(U, 1, eos, Flux);
        add_flux_divergence(Flux, 1, 1.0 / g.dy(), Rhs);
    }

    if (U.nz() > 1) {
        compute_flux_dir(U, 2, eos, Flux);
        add_flux_divergence(Flux, 2, 1.0 / g.dz(), Rhs);
    }
}

Real max_dt_hyperbolic(const State& U, const Grid& g, const IdealGas& eos,
                       Real cfl) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz();
    const auto& rho = U[RHO];
    const auto& mx  = U[RHOU];
    const auto& my  = U[RHOV];
    const auto& mz  = U[RHOW];
    const auto& E   = U[RHOE];
    const Real dx = g.dx(), dy = g.dy(), dz = g.dz();

    Real max_inv_dt = 0.0;
#pragma omp parallel for collapse(2) schedule(static) reduction(max:max_inv_dt)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const Real r  = rho(i,j,k);
                const Real iu = mx(i,j,k) / r;
                const Real iv = my(i,j,k) / r;
                const Real iw = mz(i,j,k) / r;
                const Real ke = 0.5 * r * (iu*iu + iv*iv + iw*iw);
                const Real p  = eos.pressure(r, E(i,j,k) - ke);
                const Real c  = eos.sound_speed(r, p);
                const Real lx = (std::fabs(iu) + c) / dx;
                const Real ly = (ny > 1) ? (std::fabs(iv) + c) / dy : 0.0;
                const Real lz = (nz > 1) ? (std::fabs(iw) + c) / dz : 0.0;
                const Real loc = lx + ly + lz;
                if (loc > max_inv_dt) max_inv_dt = loc;
            }
    return (max_inv_dt > 0.0) ? cfl / max_inv_dt : 1e30;
}

}  // namespace blast
