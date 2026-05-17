#pragma once

#include "core/Config.hpp"
#include "core/State.hpp"

#ifdef BLAST_MPI
#include "parallel/Domain.hpp"
#endif

namespace blast {

// Fill ghost cells on all six faces of every conserved variable in `U`.
// Velocity components are mirrored with sign-flip on the wall-normal direction
// for SlipWall faces. Periodic copies wrap. Outflow uses zero-order
// extrapolation (sufficient for sub-/super-sonic outflow over short times).
void apply_bcs(State& U, const BCSet& bc);

#ifdef BLAST_MPI
// MPI variant: same as serial apply_bcs but only fills faces that this rank
// owns physically (Domain::is_physical_face). Internal partition faces are
// expected to be filled by Halo::exchange prior to this call.
void apply_bcs(State& U, const BCSet& bc, const Domain& d);
#endif

}  // namespace blast
