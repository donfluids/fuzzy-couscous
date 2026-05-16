#pragma once

#include "core/Config.hpp"
#include "core/State.hpp"

namespace blast {

// Fill ghost cells on all six faces of every conserved variable in `U`.
// Velocity components are mirrored with sign-flip on the wall-normal direction
// for SlipWall faces. Periodic copies wrap. Outflow uses zero-order
// extrapolation (sufficient for sub-/super-sonic outflow over short times).
void apply_bcs(State& U, const BCSet& bc);

}  // namespace blast
