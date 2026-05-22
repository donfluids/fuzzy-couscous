#include "numerics/RHS.hpp"

#include "core/Field3D.hpp"
#include "numerics/ArtificialDiffusivity.hpp"
#include "numerics/CompactScheme.hpp"
#include "numerics/Ducros.hpp"
#include "numerics/Gradients.hpp"
#include "numerics/HyperdissipationSpectral.hpp"
#include "numerics/RhsScratch.hpp"
#include "numerics/Stencils.hpp"
#include "numerics/WENO5.hpp"
#include "physics/EOS.hpp"
#include "physics/EulerFlux.hpp"
#include "physics/ViscousFlux.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <omp.h>
#include <vector>

namespace blast {

namespace {

// Switch to WENO5 when the sensor exceeds this threshold anywhere in the
// face stencil. 0.65 follows Pirozzoli, Larsson, Bernardini.
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
                         State& Flux, Field3D& alpha, const Field3D* gfn) {
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
                const Real gloc = gfn ? (1.0 + 1.0 / (*gfn)(i,j,k)) : eos.eos.gamma;
                const Real p  = (gloc - 1.0) * (C.rhoE - ke);
                const Real c  = std::sqrt(gloc * p / C.rho);
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

// Dilate theta along direction d with a half-width-2 max filter.
// After this, theta_dilated(c) = max(theta_orig over [c-2, c+2] along d).
// Combined with checking both cells of a face, the central6 stencil
// (cells c-2..c+3 around the left cell) is fully covered.
//
// `scratch` is the caller's pre-allocated buffer of the same shape as theta;
// it is overwritten internally and theta is restored from it on exit.
void dilate_sensor_along(Field3D& theta, Field3D& scratch, int d) {
    const int nx = theta.nx(), ny = theta.ny(), nz = theta.nz(), ng = theta.ng();

    // Read theta on interior + ng-1 ghost layer, write into scratch.
    const int lo = -(ng - 1);
    const auto read_safe = [&](int i, int j, int k) {
        // Clamp to where theta is computed (interior + 1 ghost), elsewhere 0.
        if (i < -1 || i > nx) return 0.0;
        if (j < -1 || j > ny) return 0.0;
        if (k < -1 || k > nz) return 0.0;
        return static_cast<Real>(theta(i, j, k));
    };

#pragma omp parallel for collapse(2) schedule(static)
    for (int k = lo; k < nz + ng - 1; ++k)
        for (int j = lo; j < ny + ng - 1; ++j)
            for (int i = lo; i < nx + ng - 1; ++i) {
                Real m = 0.0;
                for (int s = -2; s <= 2; ++s) {
                    int ii = i, jj = j, kk = k;
                    if (d == 0) ii = i + s;
                    if (d == 1) jj = j + s;
                    if (d == 2) kk = k + s;
                    m = std::max(m, read_safe(ii, jj, kk));
                }
                scratch(i, j, k) = m;
            }

    // Copy scratch back into theta (whole region).
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = lo; k < nz + ng - 1; ++k)
        for (int j = lo; j < ny + ng - 1; ++j)
            for (int i = lo; i < nx + ng - 1; ++i)
                theta(i, j, k) = scratch(i, j, k);
}

// Multifluid inviscid divergence with GATED double-flux (Abgrall & Karni, JCP
// 2001). A fully conservative scheme with a per-cell variable gamma produces
// spurious pressure oscillations at a gamma-contact (Karni 1994), which blow up
// a strong products/air interface. The double-flux remedy: when updating cell c,
// FREEZE gamma = gamma_c across c's whole flux stencil so the local scheme is a
// single-gamma scheme (no contact oscillation). In a single-fluid region (G
// uniform across the stencil) the frozen-gamma flux is IDENTICAL to the cheap
// shared conservative flux, so we only pay the per-cell recompute within radius
// 3 of a contact (the `contact` flag). Bulk cells take the fast shared path.
// The shared face is evaluated twice near a contact (once per neighbour, with
// each one's gamma) -> energy is not exactly conserved across a contact, the
// accepted O(d gamma) trade-off.
void add_inviscid_divergence_mf(const State& U, const State& Flux,
                                const Field3D& alpha, const Field3D& theta,
                                const Field3D& G, const Field3D& contact,
                                int d, Real inv_dh, State& Rhs) {
    const int nx = Rhs.nx(), ny = Rhs.ny(), nz = Rhs.nz();
    const int di = (d == 0), dj = (d == 1), dk = (d == 2);
    const Index s = stride_for(d, Flux[RHO]);
    const auto& rho = U[RHO];
    const auto& mx  = U[RHOU];
    const auto& my  = U[RHOV];
    const auto& mz  = U[RHOW];
    const auto& E   = U[RHOE];

#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const int im = i - di, jm = j - dj, km = k - dk;
                const int ip = i + di, jp = j + dj, kp = k + dk;
                const Real th_lo = std::max(theta(im,jm,km), theta(i,j,k));
                const Real th_hi = std::max(theta(i,j,k),    theta(ip,jp,kp));
                const bool weno_lo = th_lo > kSensorThreshold;
                const bool weno_hi = th_hi > kSensorThreshold;

                // Is a gamma-contact within this cell's flux stencil (radius 3)?
                bool near = false;
                for (int t = -3; t <= 3; ++t)
                    if (contact(i + t*di, j + t*dj, k + t*dk) > 0.5) { near = true; break; }

                if (!near) {
                    // Fast shared-flux conservative path (== single-gas).
                    const Real alpha_lo = std::max(alpha(im,jm,km), alpha(i,j,k));
                    const Real alpha_hi = std::max(alpha(i,j,k),    alpha(ip,jp,kp));
                    for (int v = 0; v < NCONS; ++v) {
                        const Field3D& F = Flux[v];
                        const Field3D& Q = U[v];
                        const Real Fhi = face_flux(&F(i,j,k),   &Q(i,j,k),   alpha_hi, s, weno_hi);
                        const Real Flo = face_flux(&F(im,jm,km),&Q(im,jm,km),alpha_lo, s, weno_lo);
                        Rhs[v](i,j,k) -= (Fhi - Flo) * inv_dh;
                    }
                    continue;
                }

                // Double-flux: recompute the 7-cell local stencil with the
                // update cell's frozen gamma (component-major, stride 1).
                const Real gm1 = 1.0 / G(i, j, k);     // gamma - 1
                const Real gam = 1.0 + gm1;
                Real Fl[NCONS][7], Ul[NCONS][7], al[7];
                for (int t2 = -3; t2 <= 3; ++t2) {
                    const int t = t2 + 3;
                    const int ii = i + t2*di, jj = j + t2*dj, kk = k + t2*dk;
                    const Real r  = rho(ii,jj,kk);
                    const Real ru = mx(ii,jj,kk), rv = my(ii,jj,kk), rw = mz(ii,jj,kk);
                    const Real Ec = E(ii,jj,kk);
                    const Real inv = 1.0 / r;
                    const Real u = ru*inv, v = rv*inv, w = rw*inv;
                    const Real ke = 0.5*(ru*u + rv*v + rw*w);
                    const Real p  = gm1 * (Ec - ke);            // frozen-gamma p
                    const Real ud = (d == 0 ? u : d == 1 ? v : w);
                    Ul[RHO ][t] = r;   Fl[RHO ][t] = r*ud;
                    Ul[RHOU][t] = ru;  Fl[RHOU][t] = ru*ud + (d==0)*p;
                    Ul[RHOV][t] = rv;  Fl[RHOV][t] = rv*ud + (d==1)*p;
                    Ul[RHOW][t] = rw;  Fl[RHOW][t] = rw*ud + (d==2)*p;
                    Ul[RHOE][t] = Ec;  Fl[RHOE][t] = (Ec + p)*ud;
                    const Real c = std::sqrt(std::max(gam*p/r, 0.0));
                    al[t] = std::fabs(ud) + c;
                }
                const Real alpha_hi = std::max(al[3], al[4]);   // cells i, i+1
                const Real alpha_lo = std::max(al[2], al[3]);   // cells i-1, i
                for (int v = 0; v < NCONS; ++v) {
                    const Real Fhi = face_flux(&Fl[v][3], &Ul[v][3], alpha_hi, 1, weno_hi);
                    const Real Flo = face_flux(&Fl[v][2], &Ul[v][2], alpha_lo, 1, weno_lo);
                    Rhs[v](i,j,k) -= (Fhi - Flo) * inv_dh;
                }
            }
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
                    int im = (d == 0 ? i - 1 : i);
                    int jm = (d == 1 ? j - 1 : j);
                    int km = (d == 2 ? k - 1 : k);
                    int ip = (d == 0 ? i + 1 : i);
                    int jp = (d == 1 ? j + 1 : j);
                    int kp = (d == 2 ? k + 1 : k);

                    // theta has been dilated along d with half-width 2, so
                    // max over both adjacent cells covers the full central6
                    // stencil footprint.
                    const Real th_lo = std::max(theta(im, jm, km), theta(i,  j,  k ));
                    const Real th_hi = std::max(theta(i,  j,  k ), theta(ip, jp, kp));
                    const bool weno_lo = th_lo > kSensorThreshold;
                    const bool weno_hi = th_hi > kSensorThreshold;

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

// Line-oriented hybrid divergence: 10th-order conservative compact reconstruction
// in smooth regions, WENO5 at shock-flagged faces. Per line along d we solve the
// compact pentadiagonal system for all face fluxes, override shock faces with the
// Lax-Friedrichs WENO flux, then take the conservative difference. Each face
// carries exactly one value (shared by both neighbours) so conservation
// telescopes bit-for-bit, identical to the explicit central6 path.
void add_face_flux_divergence_compact(const State& U, const State& Flux,
                                      const Field3D& alpha, const Field3D& theta,
                                      int d, Real inv_h, State& Rhs,
                                      const CompactPenta& cp) {
    const int nx = Rhs.nx(), ny = Rhs.ny(), nz = Rhs.nz();
    const int nL  = (d == 0 ? nx : d == 1 ? ny : nz);
    const int nt1 = (d == 0 ? ny : nx);              // first transverse extent
    const int nt2 = (d == 0 ? nz : d == 1 ? nz : ny);  // second transverse extent
    constexpr int NG = kCompactGhost;                // gather halo

    auto IJK = [d](int m, int t1, int t2, int& i, int& j, int& k) {
        if (d == 0)      { i = m;  j = t1; k = t2; }
        else if (d == 1) { i = t1; j = m;  k = t2; }
        else             { i = t1; j = t2; k = m;  }
    };

#pragma omp parallel
    {
        std::vector<Real> Fb[NCONS], Qb[NCONS], fc[NCONS];
        for (int v = 0; v < NCONS; ++v) {
            Fb[v].resize(nL + 2 * NG);
            Qb[v].resize(nL + 2 * NG);
            fc[v].resize(nL + 1);
        }
        std::vector<Real> ab(nL + 2 * NG), tb(nL + 2 * NG);

#pragma omp for collapse(2) schedule(static)
        for (int t2 = 0; t2 < nt2; ++t2)
            for (int t1 = 0; t1 < nt1; ++t1) {
                // Gather the line (node fluxes, conserved state, alpha, sensor)
                // with NG ghosts into contiguous buffers.
                for (int m = -NG; m < nL + NG; ++m) {
                    int i, j, k; IJK(m, t1, t2, i, j, k);
                    const Index off = Flux[0].idx_(i, j, k);
                    for (int v = 0; v < NCONS; ++v) {
                        Fb[v][m + NG] = Flux[v].raw()[off];
                        Qb[v][m + NG] = U[v].raw()[off];
                    }
                    ab[m + NG] = alpha.raw()[off];
                    tb[m + NG] = theta.raw()[off];
                }
                // Compact reconstruction (node ptr = cell 0 = buf + NG).
                for (int v = 0; v < NCONS; ++v)
                    cp.reconstruct_wall(Fb[v].data() + NG, fc[v].data());
                // Override shock faces with WENO5 (LF split). Face f sits between
                // cells f-1 and f; the sensor is dilated so the cell max covers
                // the stencil footprint.
                for (int f = 0; f <= nL; ++f) {
                    const Real th = std::max(tb[NG + f - 1], tb[NG + f]);
                    if (th > kSensorThreshold) {
                        const Real af = std::max(ab[NG + f - 1], ab[NG + f]);
                        for (int v = 0; v < NCONS; ++v)
                            fc[v][f] = face_flux(Fb[v].data() + NG + (f - 1),
                                                 Qb[v].data() + NG + (f - 1),
                                                 af, 1, true);
                    }
                }
                // Conservative difference.
                for (int m = 0; m < nL; ++m) {
                    int i, j, k; IJK(m, t1, t2, i, j, k);
                    const Index off = Rhs[0].idx_(i, j, k);
                    for (int v = 0; v < NCONS; ++v)
                        Rhs[v].raw()[off] -= (fc[v][m + 1] - fc[v][m]) * inv_h;
                }
            }
    }
}

}  // namespace

void compute_rhs_inviscid(const State& U, const Grid& g, const IdealGas& eos,
                          RhsScratch& scratch, State& Rhs, const Field3D* gfn) {
    for (int v = 0; v < NCONS; ++v) Rhs.fill(v, 0.0);

    Field3D& theta     = scratch.theta;
    Field3D& alpha     = scratch.alpha;
    Field3D& theta_dil = scratch.theta_dil;
    Field3D& dil_tmp   = scratch.dilate_tmp;
    State&   Flux      = scratch.Flux_inv;

    compute_sensor(U, g, eos, theta);

    // LAD-only mode: suppress the Ducros/WENO sensor so the central scheme runs
    // everywhere and localized artificial diffusivity is the sole shock sink.
    if (scratch.disable_weno) theta.fill(0.0);

    // Multifluid: shock sensors miss CONTACTS (uniform p,u; jump in rho/gamma),
    // so the high-order central scheme rings at the products-air interface. Force
    // WENO there by firing the sensor on a relative G = 1/(gamma-1) jump, and
    // flag the cell in `contact` so the gated double-flux activates near it.
    Field3D& contact = scratch.contact;
    if (gfn) {
        const Field3D& G = *gfn;
        const int nx = U.nx(), ny = U.ny(), nz = U.nz(), ng = U.ng();
        contact.fill(0.0);
        // Compute on a band wide enough for the radius-3 stencil reads in
        // add_inviscid_divergence_mf (interior cell stencils reach +/-3).
        const int lo = -ng + 1, hix = nx + ng - 1, hiy = ny + ng - 1, hiz = nz + ng - 1;
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = lo; k < hiz; ++k)
            for (int j = lo; j < hiy; ++j)
                for (int i = lo; i < hix; ++i) {
                    const Real Gc = G(i,j,k);
                    Real m = 0.0;
                    m = std::max(m, std::fabs(G(i+1,j,k)-Gc));
                    m = std::max(m, std::fabs(G(i-1,j,k)-Gc));
                    m = std::max(m, std::fabs(G(i,j+1,k)-Gc));
                    m = std::max(m, std::fabs(G(i,j-1,k)-Gc));
                    m = std::max(m, std::fabs(G(i,j,k+1)-Gc));
                    m = std::max(m, std::fabs(G(i,j,k-1)-Gc));
                    if (m / (Gc + 1e-30) > 0.01) {
                        contact(i,j,k) = 1.0;
                        if (i >= -1 && i <= nx && j >= -1 && j <= ny && k >= -1 && k <= nz)
                            theta(i,j,k) = 1.0;
                    }
                }
    }

    auto run_direction = [&](int d, Real inv_h) {
        // Reset dilated sensor to original, then dilate along d in-place
        // using the pre-allocated scratch buffer.
        const int ng = theta.ng();
        const Index N = theta.ldx() * (theta.ny() + 2 * ng) * (theta.nz() + 2 * ng);
#pragma omp parallel for schedule(static)
        for (Index i = 0; i < N; ++i) theta_dil.raw()[i] = theta.raw()[i];
        dilate_sensor_along(theta_dil, dil_tmp, d);

        fill_flux_and_alpha(U, d, eos, Flux, alpha, gfn);
        if (gfn) {
            // Multifluid: gated double-flux. Fast shared conservative flux in
            // single-fluid cells; frozen-gamma recompute only near a contact.
            add_inviscid_divergence_mf(U, Flux, alpha, theta_dil, *gfn, contact,
                                       d, inv_h, Rhs);
        } else if (scratch.use_compact) {
            // 10th-order conservative compact reconstruction (smooth) + WENO5
            // (shocks). Build/cache the per-direction pentadiagonal solver.
            const int nL = (d == 0 ? U.nx() : d == 1 ? U.ny() : U.nz());
            auto& slot = (d == 0 ? scratch.compact_x
                                 : d == 1 ? scratch.compact_y
                                          : scratch.compact_z);
            if (!slot || slot->n() != nL) slot = std::make_unique<CompactPenta>(nL);
            add_face_flux_divergence_compact(U, Flux, alpha, theta_dil, d, inv_h,
                                             Rhs, *slot);
        } else {
            add_face_flux_divergence(U, Flux, alpha, theta_dil, d, inv_h, Rhs);
        }
    };

    run_direction(0, 1.0 / g.dx());
    if (U.ny() > 1) run_direction(1, 1.0 / g.dy());
    if (U.nz() > 1) run_direction(2, 1.0 / g.dz());
}

void compute_rhs_inviscid(const State& U, const Grid& g, const IdealGas& eos,
                          State& Rhs, const Field3D* gfn) {
    // Standalone path used by tests / ad-hoc callers. Allocates scratch
    // once on the stack; production runs go through the RhsScratch
    // overload via RK3.
    RhsScratch s;
    s.allocate(U.nx(), U.ny(), U.nz(), U.ng());
    compute_rhs_inviscid(U, g, eos, s, Rhs, gfn);
}

Real max_dt_hyperbolic(const State& U, const Grid& g, const IdealGas& eos,
                       Real cfl, const Field3D* gfn) {
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
                const Real gloc = gfn ? (1.0 + 1.0 / (*gfn)(i,j,k)) : eos.eos.gamma;
                const Real p  = (gloc - 1.0) * (E(i,j,k) - ke);
                const Real c  = std::sqrt(gloc * p / r);
                const Real lx = (std::fabs(iu) + c) / dx;
                const Real ly = (ny > 1) ? (std::fabs(iv) + c) / dy : 0.0;
                const Real lz = (nz > 1) ? (std::fabs(iw) + c) / dz : 0.0;
                const Real loc = lx + ly + lz;
                if (loc > max_inv_dt) max_inv_dt = loc;
            }
    return (max_inv_dt > 0.0) ? cfl / max_inv_dt : 1e30;
}

Real max_dt_viscous(const State& U, const Grid& g, const ViscousParams& vp,
                    Real cfl) {
    if (vp.mu <= 0.0) return 1e30;
    const int nx = U.nx(), ny = U.ny(), nz = U.nz();
    const auto& rho = U[RHO];
    const Real dx2 = g.dx() * g.dx();
    const Real dy2 = (ny > 1) ? g.dy() * g.dy() : 1e30;
    const Real dz2 = (nz > 1) ? g.dz() * g.dz() : 1e30;
    const Real h2_min = std::min({dx2, dy2, dz2});

    Real rho_min = 1e30;
#pragma omp parallel for collapse(2) schedule(static) reduction(min:rho_min)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                if (rho(i, j, k) < rho_min) rho_min = rho(i, j, k);

    const Real nu_max = vp.mu / std::max(rho_min, 1e-30);
    return cfl * h2_min / nu_max;
}

#ifdef BLAST_MPI
Real max_dt_hyperbolic(const State& U, const Grid& g, const IdealGas& eos,
                       Real cfl, MPI_Comm comm) {
    Real dt_local = max_dt_hyperbolic(U, g, eos, cfl);
    Real dt_global = dt_local;
    MPI_Allreduce(&dt_local, &dt_global, 1, MPI_DOUBLE, MPI_MIN, comm);
    return dt_global;
}

Real max_dt_viscous(const State& U, const Grid& g, const ViscousParams& vp,
                    Real cfl, MPI_Comm comm) {
    Real dt_local = max_dt_viscous(U, g, vp, cfl);
    Real dt_global = dt_local;
    MPI_Allreduce(&dt_local, &dt_global, 1, MPI_DOUBLE, MPI_MIN, comm);
    return dt_global;
}
#endif

namespace {

// Cell-centered Stokes-form viscous flux for direction d, computed from
// pre-evaluated velocity / temperature gradients.
void fill_viscous_flux_dir(const State& U, const CellGradients& G,
                           const ViscousParams& vp, const IdealGas& eos,
                           int d, State& Flux) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz(), ng = U.ng();
    // Gradients are valid on [-ng+3, n+ng-3]. We need viscous flux on the
    // same region so the next 6th-order outer derivative is well-defined
    // for interior cells.
    const int lo = -ng + stencil::RADIUS;
    const int hi_x = nx + ng - stencil::RADIUS;
    const int hi_y = ny + ng - stencil::RADIUS;
    const int hi_z = nz + ng - stencil::RADIUS;

    const auto& rho = U[RHO];
    const auto& mx  = U[RHOU];
    const auto& my  = U[RHOV];
    const auto& mz  = U[RHOW];

    Field3D& Fr = Flux[RHO];
    Field3D& Fu = Flux[RHOU];
    Field3D& Fv = Flux[RHOV];
    Field3D& Fw = Flux[RHOW];
    Field3D& FE = Flux[RHOE];

    // Zero the full padded region first; outer derivative will only read
    // interior+radius cells, but cleaner to start with zeroed scratch.
    Fr.fill(0.0); Fu.fill(0.0); Fv.fill(0.0); Fw.fill(0.0); FE.fill(0.0);

#pragma omp parallel for collapse(2) schedule(static)
    for (int k = lo; k < hi_z; ++k)
        for (int j = lo; j < hi_y; ++j)
            for (int i = lo; i < hi_x; ++i) {
                const Real r  = rho(i, j, k);
                CellState C{};
                C.u = mx(i, j, k) / r;
                C.v = my(i, j, k) / r;
                C.w = mz(i, j, k) / r;
                // T not used directly in flux assembly but kept for completeness.
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) C.dudx[a][b] = G.du[a][b](i, j, k);
                    C.dTdx[a] = G.dT[a](i, j, k);
                }
                ViscousFluxVec Gv = viscous_flux(C, vp, eos, d);
                Fr(i, j, k) = Gv.f[RHO ];
                Fu(i, j, k) = Gv.f[RHOU];
                Fv(i, j, k) = Gv.f[RHOV];
                Fw(i, j, k) = Gv.f[RHOW];
                FE(i, j, k) = Gv.f[RHOE];
            }
}

// Cell-centered artificial (LAD) flux for direction d, on [-1, n+1) -- exactly
// the band the compact (2nd-order) divergence below needs for interior cells.
void fill_artificial_flux_dir(const State& U, const CellGradients& G,
                              const Field3D& mu_art, const Field3D& beta_art,
                              const Field3D& kappa_art, const Field3D& d_art,
                              int d, Real inv_dh, State& Flux) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz();
    const int lo = -1;
    const int hi_x = nx + 1;
    const int hi_y = (ny > 1) ? ny + 1 : ny;
    const int hi_z = (nz > 1) ? nz + 1 : nz;
    const int jlo = (ny > 1) ? -1 : 0;
    const int klo = (nz > 1) ? -1 : 0;

    const auto& rho = U[RHO];
    const auto& mx  = U[RHOU];
    const auto& my  = U[RHOV];
    const auto& mz  = U[RHOW];
    Field3D& Fr = Flux[RHO]; Field3D& Fu = Flux[RHOU]; Field3D& Fv = Flux[RHOV];
    Field3D& Fw = Flux[RHOW]; Field3D& FE = Flux[RHOE];
    Fr.fill(0.0); Fu.fill(0.0); Fv.fill(0.0); Fw.fill(0.0); FE.fill(0.0);

    const Index srho = (d == 0 ? 1 : (d == 1 ? rho.ldx() : rho.ldxy()));
    const Real half_inv = 0.5 * inv_dh;
    auto zsafe = [](Real x) { return std::isfinite(x) ? x : 0.0; };

#pragma omp parallel for collapse(2) schedule(static)
    for (int k = klo; k < hi_z; ++k)
        for (int j = jlo; j < hi_y; ++j)
            for (int i = lo; i < hi_x; ++i) {
                const Real r = rho(i, j, k);
                CellState C{};
                C.u = mx(i, j, k) / r;
                C.v = my(i, j, k) / r;
                C.w = mz(i, j, k) / r;
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) C.dudx[a][b] = G.du[a][b](i, j, k);
                    C.dTdx[a] = G.dT[a](i, j, k);
                }
                ViscousFluxVec Gv = artificial_flux(
                    C, mu_art(i, j, k), beta_art(i, j, k), kappa_art(i, j, k), d);

                // Consistent mass/contact diffusion: J = D_art * d(rho)/dx_d.
                // Diffusing rho carries momentum at the local velocity and the
                // local kinetic energy, so u and (internal energy hence) p are
                // unchanged across a contact -- only rho is smoothed.
                const Real* rp = &rho(i, j, k);
                const Real drho = (zsafe(rp[srho]) - zsafe(rp[-srho])) * half_inv;
                const Real J = d_art(i, j, k) * drho;
                const Real ke = 0.5 * (C.u * C.u + C.v * C.v + C.w * C.w);

                Fr(i, j, k) = Gv.f[RHO ] + J;
                Fu(i, j, k) = Gv.f[RHOU] + J * C.u;
                Fv(i, j, k) = Gv.f[RHOV] + J * C.v;
                Fw(i, j, k) = Gv.f[RHOW] + J * C.w;
                FE(i, j, k) = Gv.f[RHOE] + J * ke;
            }
}

// Compact 2nd-order central divergence of an artificial flux (radius 1), so it
// needs the flux only on [-1, n+1). The artificial diffusivity is a sub-grid
// regularization, so a low-order divergence is appropriate and keeps the
// 2nd-derivative LAD sensor within NGHOST=6.
void add_compact_divergence(const State& Flux, int d, Real inv_dh, State& Rhs) {
    const int nx = Rhs.nx(), ny = Rhs.ny(), nz = Rhs.nz();
    const Real half_inv = 0.5 * inv_dh;
    // Includes RHO: the artificial mass/contact diffusion has a non-zero mass
    // flux (the physical viscous flux does not).
    for (int v = 0; v < NCONS; ++v) {
        const Field3D& F = Flux[v];
        Field3D&       R = Rhs[v];
        const Index sd = (d == 0 ? 1 : (d == 1 ? F.ldx() : F.ldxy()));
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
#pragma omp simd
                for (int i = 0; i < nx; ++i) {
                    const Real* f = &F(i, j, k);
                    R(i, j, k) += (f[sd] - f[-sd]) * half_inv;
                }
    }
}

void add_viscous_flux_divergence(const State& Flux, int d, Real inv_dh,
                                 State& Rhs) {
    const int nx = Rhs.nx(), ny = Rhs.ny(), nz = Rhs.nz();
    const Index s = (d == 0 ? 1
                   : d == 1 ? Flux[RHO].ldx()
                            : Flux[RHO].ldx() * Flux[RHO].ldxy() / Flux[RHO].ldx());
    for (int v = 0; v < NCONS; ++v) {
        if (v == RHO) continue;  // viscous flux is zero for mass eqn
        const Field3D& F = Flux[v];
        Field3D&       R = Rhs[v];
        const Index sd = (d == 0 ? 1 : (d == 1 ? F.ldx() : F.ldxy()));
        (void)s;
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
#pragma omp simd
                for (int i = 0; i < nx; ++i) {
                    R(i, j, k) += stencil::ddx_6(&F(i, j, k), sd, inv_dh);
                }
    }
}

}  // namespace

// Composed (nabla^2)^2 applied per conserved variable as -nu_h * Lap(Lap(U)).
// Pass 1 evaluates the Laplacian into a scratch field on interior + 3 ghost
// rings; pass 2 takes the Laplacian of that and ADDS the contribution.
// Requires NGHOST >= 6 (we have it).
static void add_rhs_hyperdissipation(const State& U, const Grid& g,
                                     Real nu_h, Field3D& lap, State& Rhs) {
    if (nu_h <= 0.0) return;
    const int nx = U.nx(), ny = U.ny(), nz = U.nz(), ng = U.ng();
    const Real inv_dx2 = 1.0 / (g.dx() * g.dx());
    const Real inv_dy2 = (ny > 1) ? 1.0 / (g.dy() * g.dy()) : 0.0;
    const Real inv_dz2 = (nz > 1) ? 1.0 / (g.dz() * g.dz()) : 0.0;
    const Index sx = 1;

    const int lo = -ng + stencil::RADIUS;
    const int hi_x = nx + ng - stencil::RADIUS;
    const int hi_y = ny + ng - stencil::RADIUS;
    const int hi_z = nz + ng - stencil::RADIUS;

    for (int v = 0; v < NCONS; ++v) {
        const Field3D& Uv = U[v];
        const Index sy = Uv.ldx();
        const Index sz = Uv.ldxy();

#pragma omp parallel for collapse(2) schedule(static)
        for (int k = lo; k < hi_z; ++k)
            for (int j = lo; j < hi_y; ++j) {
#pragma omp simd
                for (int i = lo; i < hi_x; ++i) {
                    const Real* p = &Uv(i, j, k);
                    Real L = stencil::d2dx2_6(p, sx, inv_dx2);
                    if (ny > 1) L += stencil::d2dx2_6(p, sy, inv_dy2);
                    if (nz > 1) L += stencil::d2dx2_6(p, sz, inv_dz2);
                    lap(i, j, k) = L;
                }
            }

        Field3D& Rv = Rhs[v];
        const Index lsx = 1, lsy = lap.ldx(), lsz = lap.ldxy();
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j) {
#pragma omp simd
                for (int i = 0; i < nx; ++i) {
                    const Real* q = &lap(i, j, k);
                    Real L2 = stencil::d2dx2_6(q, lsx, inv_dx2);
                    if (ny > 1) L2 += stencil::d2dx2_6(q, lsy, inv_dy2);
                    if (nz > 1) L2 += stencil::d2dx2_6(q, lsz, inv_dz2);
                    Rv(i, j, k) -= nu_h * L2;
                }
            }
    }
}

// Composed (nabla^2)^3 applied per conserved variable as +nu_h6 * Lap^3(U).
// Sign flips relative to nabla^4: (nabla^2)^n sin(kx) = (-k^2)^n sin(kx), so
// to get the decay rate -nu_{2n} k^{2n} we use coefficient (-1)^{n+1}; n=3
// gives +1.
//
// Pass 1: lap  = nabla^2 U  on interior + 4 ghost rings.
// Pass 2: lap2 = nabla^2 lap on interior + 2 ghost rings.
// Pass 3: nabla^2 lap2 on interior, ADDED to RHS with sign +nu_h6.
// Each pass uses the radius-2 (4th-order accurate) Laplacian d2dx2_4 so
// 3 * 2 = 6 ghost cells suffice. NGHOST = 6 is exactly used.
static void add_rhs_hyperdissipation6(const State& U, const Grid& g,
                                      Real nu_h6, Field3D& lap, Field3D& lap2,
                                      State& Rhs) {
    if (nu_h6 <= 0.0) return;
    const int nx = U.nx(), ny = U.ny(), nz = U.nz(), ng = U.ng();
    const Real inv_dx2 = 1.0 / (g.dx() * g.dx());
    const Real inv_dy2 = (ny > 1) ? 1.0 / (g.dy() * g.dy()) : 0.0;
    const Real inv_dz2 = (nz > 1) ? 1.0 / (g.dz() * g.dz()) : 0.0;
    constexpr int R = 2;
    const Index sx = 1;

    const int lo1 = -ng + R,         hi1_x = nx + ng - R,
                                     hi1_y = ny + ng - R,
                                     hi1_z = nz + ng - R;
    const int lo2 = -ng + 2 * R,     hi2_x = nx + ng - 2 * R,
                                     hi2_y = ny + ng - 2 * R,
                                     hi2_z = nz + ng - 2 * R;

    for (int v = 0; v < NCONS; ++v) {
        const Field3D& Uv = U[v];
        const Index sy = Uv.ldx();
        const Index sz = Uv.ldxy();
        const Index lsx = 1, lsy = lap.ldx(), lsz = lap.ldxy();
        const Index l2sx = 1, l2sy = lap2.ldx(), l2sz = lap2.ldxy();

        // PASS 1: lap = nabla^2 U on [-4, n+4)
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = lo1; k < hi1_z; ++k)
            for (int j = lo1; j < hi1_y; ++j) {
#pragma omp simd
                for (int i = lo1; i < hi1_x; ++i) {
                    const Real* p = &Uv(i, j, k);
                    Real L = stencil::d2dx2_4(p, sx, inv_dx2);
                    if (ny > 1) L += stencil::d2dx2_4(p, sy, inv_dy2);
                    if (nz > 1) L += stencil::d2dx2_4(p, sz, inv_dz2);
                    lap(i, j, k) = L;
                }
            }

        // PASS 2: lap2 = nabla^2 lap on [-2, n+2)
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = lo2; k < hi2_z; ++k)
            for (int j = lo2; j < hi2_y; ++j) {
#pragma omp simd
                for (int i = lo2; i < hi2_x; ++i) {
                    const Real* q = &lap(i, j, k);
                    Real L = stencil::d2dx2_4(q, lsx, inv_dx2);
                    if (ny > 1) L += stencil::d2dx2_4(q, lsy, inv_dy2);
                    if (nz > 1) L += stencil::d2dx2_4(q, lsz, inv_dz2);
                    lap2(i, j, k) = L;
                }
            }

        Field3D& Rv = Rhs[v];
        // PASS 3: Rv += nu_h6 * nabla^2 lap2 on [0, n)
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j) {
#pragma omp simd
                for (int i = 0; i < nx; ++i) {
                    const Real* r = &lap2(i, j, k);
                    Real L3 = stencil::d2dx2_4(r, l2sx, inv_dx2);
                    if (ny > 1) L3 += stencil::d2dx2_4(r, l2sy, inv_dy2);
                    if (nz > 1) L3 += stencil::d2dx2_4(r, l2sz, inv_dz2);
                    Rv(i, j, k) += nu_h6 * L3;
                }
            }
    }
}

void add_rhs_viscous(const State& U, const Grid& g, const IdealGas& eos,
                     const ViscousParams& vp, RhsScratch& scratch, State& Rhs) {
    // Cell gradients feed both the physical viscous flux and the LAD sensor.
    const bool need_grad = (vp.mu > 0.0) || vp.abv_enabled;
    if (need_grad) {
        compute_cell_gradients(U, g, eos,
                               scratch.prim_u, scratch.prim_v,
                               scratch.prim_w, scratch.prim_T,
                               scratch.G);
    }

    if (vp.mu > 0.0) {
        State& Flux = scratch.Flux_visc;

        fill_viscous_flux_dir(U, scratch.G, vp, eos, 0, Flux);
        add_viscous_flux_divergence(Flux, 0, 1.0 / g.dx(), Rhs);

        if (U.ny() > 1) {
            fill_viscous_flux_dir(U, scratch.G, vp, eos, 1, Flux);
            add_viscous_flux_divergence(Flux, 1, 1.0 / g.dy(), Rhs);
        }
        if (U.nz() > 1) {
            fill_viscous_flux_dir(U, scratch.G, vp, eos, 2, Flux);
            add_viscous_flux_divergence(Flux, 2, 1.0 / g.dz(), Rhs);
        }
    }

    // Localized artificial diffusivity (LAD): compute the coefficient fields
    // then add their compact-divergence flux. Shares the cell gradients above.
    if (vp.abv_enabled) {
        scratch.allocate_abv(U.nx(), U.ny(), U.nz(), U.ng());
        scratch.abv_nu_max = compute_lad_fields(
            U, g, eos, vp, scratch.G, scratch.prim_T,
            scratch.lad_theta, scratch.lad_strain,
            scratch.mu_art, scratch.beta_art, scratch.kappa_art, scratch.d_art);

        State& Flux = scratch.Flux_visc;
        const Real idx = 1.0 / g.dx(), idy = 1.0 / g.dy(), idz = 1.0 / g.dz();
        fill_artificial_flux_dir(U, scratch.G, scratch.mu_art, scratch.beta_art,
                                 scratch.kappa_art, scratch.d_art, 0, idx, Flux);
        add_compact_divergence(Flux, 0, idx, Rhs);
        if (U.ny() > 1) {
            fill_artificial_flux_dir(U, scratch.G, scratch.mu_art, scratch.beta_art,
                                     scratch.kappa_art, scratch.d_art, 1, idy, Flux);
            add_compact_divergence(Flux, 1, idy, Rhs);
        }
        if (U.nz() > 1) {
            fill_artificial_flux_dir(U, scratch.G, scratch.mu_art, scratch.beta_art,
                                     scratch.kappa_art, scratch.d_art, 2, idz, Flux);
            add_compact_divergence(Flux, 2, idz, Rhs);
        }
    }


    if (vp.hyper_method == HyperMethod::FiniteDifference) {
        add_rhs_hyperdissipation(U, g, vp.hyper_coeff, scratch.lap, Rhs);
        add_rhs_hyperdissipation6(U, g, vp.hyper6_coeff,
                                  scratch.lap, scratch.lap2, Rhs);
    } else {
        if (!scratch.spectral_hyper) {
            scratch.spectral_hyper = std::make_unique<HyperdissipationSpectral>(
                U.nx(), U.ny(), U.nz(), vp.spectral_bc_mode);
        }
        scratch.spectral_hyper->apply(U, g, vp.hyper_coeff, vp.hyper6_coeff, Rhs);
    }
}

void add_rhs_viscous(const State& U, const Grid& g, const IdealGas& eos,
                     const ViscousParams& vp, State& Rhs) {
    // Standalone wrapper for tests / ad-hoc callers.
    RhsScratch s;
    s.allocate(U.nx(), U.ny(), U.nz(), U.ng());
    add_rhs_viscous(U, g, eos, vp, s, Rhs);
}

}  // namespace blast
