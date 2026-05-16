#pragma once

#include "bc/BC.hpp"
#include "core/Config.hpp"
#include "core/State.hpp"

namespace blast {

// In-place application of the 6th-order Lele explicit low-pass filter (see
// numerics/Stencils.hpp::filter_6) sweeping x then y then z, blending with
// the unfiltered field by `sigma`. Used as a post-RK3 stability device for
// under-resolved LES: knocks down the highest 1/4 of the resolved wavenumber
// range while leaving k < k_max / 2 essentially untouched.
//
// Applies on the interior (relies on BC-filled ghosts for the radius-3
// stencil). Caller is responsible for re-applying BCs afterward.
void apply_lele_filter(State& U, const BCSet& bc, Real sigma);

}  // namespace blast
