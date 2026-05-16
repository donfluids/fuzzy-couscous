#pragma once

#include "core/Field3D.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "physics/EOS.hpp"

namespace blast {

// Composite shock sensor:
//   theta = max( Ducros_velocity, pressure_jump )
//
// Ducros et al. (1999):
//   phi_v = (div u)^2 / ((div u)^2 + |omega|^2 + eps)
//
// Pressure-jump indicator (Larsson-style addition; catches static thermodynamic
// discontinuities that velocity-based sensors miss, e.g. at t=0 of Sedov):
//   phi_p = max over 6 face neighbors of |p_n - p_c| / (p_n + p_c)
//
// Both phi_v and phi_p live in [0,1]. The hybrid scheme switches to WENO
// where theta exceeds ~0.65.
void compute_sensor(const State& U, const Grid& g, const IdealGas& eos,
                    Field3D& theta);

}  // namespace blast
