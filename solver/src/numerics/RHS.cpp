#include "numerics/RHS.hpp"

#include "core/Field3D.hpp"
#include "numerics/Ducros.hpp"
#include "numerics/WENO5.hpp"
#include "physics/EOS.hpp"
#include "physics/EulerFlux.hpp"

#include <algorithm>
#include <cmath>
#include <omp.h>

namespace blast {

namespace {

// Ducros activation threshold: switch to WENO5 when sensor exceeds this.
constexpr Real kSensorThreshold = 0.65;

// Per-cell scratch: 5 conserved-flux components evaluated at cell centers,
// plus the local maximum eigenvalue alpha = |u_d| + c used for LF splitting.
struct PerDirScratch {
    State Flux;
    Field3D alpha;
    PerDirScratch(int nx, int ny, int nz, int ng) : Flux(nx, ny, nz, ng), alpha(nx, ny, nz, ng) {}
};

Index stride_for(int d, const Field3D& f) {
    if (d == 0) return 1;
    if (d == 1) return f.ldx();
    return f.ldxy();
}

void fill_flux_and_alpha(const State& U, int d, const IdealGas& eos,
                         State& Flux, Field3D& alpha) {
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
                const Real c  = eos.sound_speed(C.rho, p);
                const Real ud = (d == 0 ? u : d == 1 ? v : w);
                FluxVec F = euler_flux(C, p, d);
                Fr(i,j,k) = F.f[RHO];
                Fu(i,j,k) = F.f[RHOU];
                Fv(i,j,k) = F.f[RHOV];
                Fw(i,j,k) = F.f[RHOW];
                FE(i,j,k) = F.f[RHOE];
                alpha(i,j,k) = std::fabs(ud) + c;
            }
        }
}

// Reconstruct numerical flux at face i+1/2 (cell-pointer + stride).
// In smooth regions, use central6 on the cell-centered flux directly.
// In shock regions, Lax-Friedrichs split + WENO5 on each half-flux.
inline Real face_flux(const Real* Fptr, const Real* Uptr, Real alpha_face,
                      Index stride, bool use_weno) {
    if (!use_weno) {
        return weno5::reconstruct_central6(Fptr, stride);
    }
    // Local Lax-Friedrichs: f+ = (F + alpha U)/2; f- = (F - alpha U)/2.
    // Reconstruct f+ from left-biased stencil, f- from right-biased.
    Real fp[5];
    Real fm[5];
    // Indices relative to face: cells i-2..i+3 (6 cells, 5 each side overlap).
    for (int s = -2; s <= 2; ++s) {
        const Real F = Fptr[s * stride];
        const Real Q = Uptr[s * stride];
        fp[s + 2] = 0.5 * (F + alpha_face * Q);
    }
    for (int s = -1; s <= 3; ++s) {
        const Real F = Fptr[s * stride];
        const Real Q = Uptr[s * stride];
        fm[s + 1] = 0.5 * (F - alpha_face * Q);
    }
    // Place pointers at center index for reconstruct calls (operate at stride=1
    // over a local array of 5 entries; we pass a contiguous buffer).
    auto recon_left = [](const Real* v) {
        // mimic weno5::reconstruct_left with stride=1 over a 5-array centered at index 2.
        const Real vm2 = v[0], vm1 = v[1], v0 = v[2], vp1 = v[3], vp2 = v[4];
        const Real d_m2_m1 = vm2 - 2.0 * vm1 + v0;
        const Real d_m2_0  = vm2 - 4.0 * vm1 + 3.0 * v0;
        const Real b0 = (13.0 / 12.0) * d_m2_m1 * d_m2_m1
                      + 0.25 * d_m2_0 * d_m2_0;
        const Real d_m1_0 = vm1 - 2.0 * v0 + vp1;
        const Real d_m1_1 = vm1 - vp1;
        const Real b1 = (13.0 / 12.0) * d_m1_0 * d_m1_0
                      + 0.25 * d_m1_1 * d_m1_1;
        const Real d_0_1  = v0 - 2.0 * vp1 + vp2;
        const Real d_0_p2 = 3.0 * v0 - 4.0 * vp1 + vp2;
        const Real b2 = (13.0 / 12.0) * d_0_1 * d_0_1
                      + 0.25 * d_0_p2 * d_0_p2;
        constexpr Real EPS_W = 1e-6;
        const Real a0 = (1.0 / 10.0) / ((EPS_W + b0) * (EPS_W + b0));
        const Real a1 = (6.0 / 10.0) / ((EPS_W + b1) * (EPS_W + b1));
        const Real a2 = (3.0 / 10.0) / ((EPS_W + b2) * (EPS_W + b2));
        const Real s = 1.0 / (a0 + a1 + a2);
        const Real p0 = ( 2.0 * vm2 - 7.0 * vm1 + 11.0 * v0 ) / 6.0;
        const Real p1 = (-1.0 * vm1 + 5.0 * v0  +  2.0 * vp1) / 6.0;
        const Real p2 = ( 2.0 * v0  + 5.0 * vp1 -  1.0 * vp2) / 6.0;
        return s * (a0 * p0 + a1 * p1 + a2 * p2);
    };
    // For f- we run the same left-biased reconstruction on the mirrored
    // stencil (cells i+3, i+2, i+1, i, i-1).
    Real fm_mirrored[5] = {fm[4], fm[3], fm[2], fm[1], fm[0]};
    const Real hp = recon_left(fp);
    const Real hm = recon_left(fm_mirrored);
    return hp + hm;
}

void add_face_flux_divergence(const State& U, const State& Flux,
                              const Field3D& alpha, const Field3D& theta,
                              int d, Real inv_dh, State& Rhs) {
    const int nx = Rhs.nx(), ny = Rhs.ny(), nz = Rhs.nz();
    for (int v = 0; v < NCONS; ++v) {
        const Field3D& F = Flux[v];
        const Field3D& Q = U[v];
        Field3D&       R = Rhs[v];
        const Index s = stride_for(d, F);

#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    // Sensor at face i-1/2: max of theta on the two cells.
                    int im = (d == 0 ? i - 1 : i);
                    int jm = (d == 1 ? j - 1 : j);
                    int km = (d == 2 ? k - 1 : k);
                    const Real th_lo_minus = theta(im, jm, km);
                    const Real th_lo_plus  = theta(i,  j,  k );
                    const bool weno_lo = (std::max(th_lo_minus, th_lo_plus) > kSensorThreshold);

                    // Sensor at face i+1/2.
                    int ip = (d == 0 ? i + 1 : i);
                    int jp = (d == 1 ? j + 1 : j);
                    int kp = (d == 2 ? k + 1 : k);
                    const Real th_hi_minus = theta(i,  j,  k );
                    const Real th_hi_plus  = theta(ip, jp, kp);
                    const bool weno_hi = (std::max(th_hi_minus, th_hi_plus) > kSensorThreshold);

                    const Real alpha_lo = std::max(alpha(im, jm, km), alpha(i,  j,  k ));
                    const Real alpha_hi = std::max(alpha(i,  j,  k ), alpha(ip, jp, kp));

                    // Cell pointers for face i+1/2 sit at cell i; for face i-1/2 at i-1.
                    const Real* Fptr_hi = &F(i, j, k);
                    const Real* Qptr_hi = &Q(i, j, k);
                    const Real* Fptr_lo = &F(im, jm, km);
                    const Real* Qptr_lo = &Q(im, jm, km);

                    const Real Fhi = face_flux(Fptr_hi, Qptr_hi, alpha_hi, s, weno_hi);
                    const Real Flo = face_flux(Fptr_lo, Qptr_lo, alpha_lo, s, weno_lo);

                    R(i, j, k) -= (Fhi - Flo) * inv_dh;
                }
            }
    }
}

}  // namespace

void compute_rhs_inviscid(const State& U, const Grid& g, const IdealGas& eos,
                          State& Rhs) {
    for (int v = 0; v < NCONS; ++v) Rhs.fill(v, 0.0);

    Field3D theta(U.nx(), U.ny(), U.nz(), U.ng());
    compute_ducros(U, g, theta);

    State Flux(U.nx(), U.ny(), U.nz(), U.ng());
    Field3D alpha(U.nx(), U.ny(), U.nz(), U.ng());

    fill_flux_and_alpha(U, 0, eos, Flux, alpha);
    add_face_flux_divergence(U, Flux, alpha, theta, 0, 1.0 / g.dx(), Rhs);

    if (U.ny() > 1) {
        fill_flux_and_alpha(U, 1, eos, Flux, alpha);
        add_face_flux_divergence(U, Flux, alpha, theta, 1, 1.0 / g.dy(), Rhs);
    }
    if (U.nz() > 1) {
        fill_flux_and_alpha(U, 2, eos, Flux, alpha);
        add_face_flux_divergence(U, Flux, alpha, theta, 2, 1.0 / g.dz(), Rhs);
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
