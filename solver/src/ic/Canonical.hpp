#pragma once

#include "core/Config.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "physics/EOS.hpp"

#include <limits>

namespace blast {

// Set U from primitive (rho, u, v, w, p).
void set_from_primitive(State& U, int i, int j, int k, const IdealGas& eos,
                        Real rho, Real u, Real v, Real w, Real p);

void ic_sod_x(State& U, const Grid& g, const IdealGas& eos);
void ic_shu_osher_x(State& U, const Grid& g, const IdealGas& eos);

// 3D Sedov-Taylor point blast: uniform pressure in a sphere of radius
// `r_blast` such that the total deposited internal energy equals `E_total`,
// surrounded by quiescent ambient (rho_ambient, p_ambient).
void ic_sedov_3d(State& U, const Grid& g, const IdealGas& eos,
                 Real E_total, Real rho_ambient, Real p_ambient,
                 Real r_blast);

// Canonical Taylor-Green vortex initial condition on a periodic cube of side
// L = 2 pi. V0 = peak velocity, M0 = reference Mach so that p_0 =
// rho_0 V0^2 / (gamma M0^2). For the Re=1600 LES benchmark use V0=1,
// rho_0=1, M0=0.1, mu = 1/1600.
void ic_taylor_green_3d(State& U, const Grid& g, const IdealGas& eos,
                        Real V0, Real rho_0, Real M_0);

// Rogallo-style divergence-free random velocity field on a periodic cube,
// with a prescribed initial energy spectrum:
//
//   E(k) = A k^4 exp(-2 (k/k_peak)^2)              (Passot-Pouquet)
//
// where A is chosen so that the integrated kinetic energy density equals
// 0.5 * urms^2. The random field is built in Fourier space with two
// independent Gaussian-random helicity modes per (kx,ky,kz) and then
// projected onto the plane perpendicular to k (divergence-free). Returns
// a quiescent thermodynamic state at (rho_0, p_0) so the flow is initially
// incompressible-like (low-Mach).
//
// Used for the canonical Comte-Bellot--Corrsin (CBC) decaying-isotropic-
// turbulence validation: with k_peak ~ 4 and urms ~ 1 on a (2 pi)^3 domain,
// the resulting decay rate and three-time spectra match the published
// experiment.
void ic_rogallo_3d(State& U, const Grid& g, const IdealGas& eos,
                   Real urms, Real k_peak, Real rho_0, Real p_0,
                   int seed);

// Chamber blast: dense hot sphere of radius r_blast inside a quiescent box.
// `tanh_thickness > 0` smooths the sphere interface (avoids initial WENO
// transients). `Y42_amp != 0` perturbs the sphere radius with the
// Y_{4,2} real spherical harmonic.
void ic_sphere_blast_3d(State& U, const Grid& g, const IdealGas& eos,
                        Real rho_blast, Real T_blast,
                        Real rho_ambient, Real T_ambient,
                        Real r_blast, Real tanh_thickness, Real Y42_amp,
                        Real ensemble_amp = 0.0, int ensemble_seed = 0,
                        // Explicit sphere center; if NaN, the local grid
                        // center g.x0+lx/2 is used. MPI callers must pass
                        // the GLOBAL center so every rank places the
                        // sphere at the same physical location.
                        Real x_center = std::numeric_limits<Real>::quiet_NaN(),
                        Real y_center = std::numeric_limits<Real>::quiet_NaN(),
                        Real z_center = std::numeric_limits<Real>::quiet_NaN());

// Energy-conserving Gaussian blast: deposit a total internal energy `E_total`
// into a C-infinity Gaussian profile of width `sigma` on a quiescent, uniform-
// density ambient (rho_ambient, T_ambient), u = 0:
//
//   (rho e)(r) = p_ambient/(gamma-1) + A exp(-r^2 / 2 sigma^2),
//   A = E_total / (2 pi sigma^2)^{3/2}   ->   integral of the bump = E_total.
//
// The blast shock forms self-consistently from the smooth pressure gradient
// (no imposed discontinuity, no density contact), so the high-order central +
// artificial-diffusivity scheme has nothing to overshoot on -- a theoretically
// grounded, fully smooth replacement for the tanh hot-sphere. `Y42_amp` and the
// ensemble perturbation modulate the deposited energy angularly (seed for
// turbulence); their angular mean is ~0 so E_total is preserved.
void ic_gaussian_blast_3d(State& U, const Grid& g, const IdealGas& eos,
                          Real E_total, Real sigma,
                          Real rho_ambient, Real T_ambient, Real Y42_amp,
                          Real ensemble_amp = 0.0, int ensemble_seed = 0,
                          Real x_center = std::numeric_limits<Real>::quiet_NaN(),
                          Real y_center = std::numeric_limits<Real>::quiet_NaN(),
                          Real z_center = std::numeric_limits<Real>::quiet_NaN());

// Chapman-Jouguet detonation initial condition (addresses reviewer M9).
// At t=0 a spherical region of radius `r_cj` contains gas in the CJ state
// behind a strong detonation wave traveling outward with detonation velocity
// D_cj. Outside is ambient (rho_0, T_0). CJ relations for an ideal gas with
// heat release q per unit ambient mass:
//   D_cj = sqrt(2 (gamma^2 - 1) q + c_0^2)        // strong detonation limit q >> c_0^2
//   p_cj / p_0  =  (1 + gamma D_cj^2/(R T_0)) / (gamma + 1)
//   rho_cj / rho_0 = (gamma + 1) * gamma * D_cj^2 / (gamma D_cj^2 + c_0^2)
//   u_cj (radial outward) = D_cj * (rho_cj - rho_0) / rho_cj
// Radial velocity behind the shock makes this much closer to a real HE
// release than the top-hat / smoothed pressure sphere.
void ic_cj_detonation_3d(State& U, const Grid& g, const IdealGas& eos,
                         Real rho_0, Real T_0, Real q_specific,
                         Real r_cj, Real tanh_thickness, Real Y42_amp,
                         Real x_center = std::numeric_limits<Real>::quiet_NaN(),
                         Real y_center = std::numeric_limits<Real>::quiet_NaN(),
                         Real z_center = std::numeric_limits<Real>::quiet_NaN());

// Smooth density wave for spatial-order verification:
//   rho = 1 + A sin(2 pi k x), u = u0, p = 1.
void ic_density_wave_x(State& U, const Grid& g, const IdealGas& eos,
                       Real amplitude, Real kwave, Real u0);

// 3D convecting entropy wave (MMS-A): exact solution of compressible
// inviscid Euler. ρ varies as a triple-sine product; (u, v, w) and p are
// uniform. The wave translates rigidly at (u0, v0, w0); at t = L / u_i the
// solution returns to its initial state. Fills interior + ghost cells.
void ic_entropy_wave_3d(State& U, const Grid& g, const IdealGas& eos,
                        Real amplitude, Real u0, Real v0, Real w0,
                        Real rho_0, Real p_0);

// Isentropic Euler vortex (Yee-Sandham-Djomehri 1999): smooth coherent 2D
// vortex of unit characteristic scale superposed on a uniform flow, embedded
// z-invariant in the 3D domain. Exact solution of compressible inviscid
// Euler; the vortex translates at (u_inf, v_inf) without distortion.
//   T  = T_inf - (gamma-1) eps^2 / (8 gamma pi^2) exp(1 - r^2)
//   rho = (T/T_inf)^(1/(gamma-1))   p = (T/T_inf)^(gamma/(gamma-1))
// with r^2 = (x - x_c)^2 + (y - y_c)^2. T_inf = rho_inf = p_inf = 1.
// Fills interior + ghost cells.
void ic_isentropic_vortex(State& U, const Grid& g, const IdealGas& eos,
                          Real eps, Real u_inf, Real v_inf,
                          Real x_c, Real y_c);

}  // namespace blast
