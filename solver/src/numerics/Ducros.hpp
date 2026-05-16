#pragma once

#include "core/Field3D.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"

namespace blast {

// Ducros et al. (1999) shock sensor:
//   theta = (div u)^2 / ((div u)^2 + |omega|^2 + eps)
// theta in [0,1]; near 1 in compressed/divergent regions, near 0 in pure
// vortex motion. Threshold ~ 0.65 is community-standard.
//
// We compute theta at cell centers using central second-order differences of
// velocity; output Field3D `theta` must be allocated with at least 1 ghost
// (for downstream face-side comparisons). Ghost layer of U must be filled.
void compute_ducros(const State& U, const Grid& g, Field3D& theta);

}  // namespace blast
