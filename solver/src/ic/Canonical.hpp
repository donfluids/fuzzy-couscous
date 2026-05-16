#pragma once

#include "core/Config.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "physics/EOS.hpp"

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

// Chamber blast: dense hot sphere of radius r_blast inside a quiescent box.
// `tanh_thickness > 0` smooths the sphere interface (avoids initial WENO
// transients). `Y42_amp != 0` perturbs the sphere radius with the
// Y_{4,2} real spherical harmonic.
void ic_sphere_blast_3d(State& U, const Grid& g, const IdealGas& eos,
                        Real rho_blast, Real T_blast,
                        Real rho_ambient, Real T_ambient,
                        Real r_blast, Real tanh_thickness, Real Y42_amp);

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
                         Real r_cj, Real tanh_thickness, Real Y42_amp);

// Smooth density wave for spatial-order verification:
//   rho = 1 + A sin(2 pi k x), u = u0, p = 1.
void ic_density_wave_x(State& U, const Grid& g, const IdealGas& eos,
                       Real amplitude, Real kwave, Real u0);

}  // namespace blast
