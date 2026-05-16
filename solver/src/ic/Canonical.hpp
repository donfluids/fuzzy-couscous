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

// Smooth density wave for spatial-order verification:
//   rho = 1 + A sin(2 pi k x), u = u0, p = 1.
void ic_density_wave_x(State& U, const Grid& g, const IdealGas& eos,
                       Real amplitude, Real kwave, Real u0);

}  // namespace blast
