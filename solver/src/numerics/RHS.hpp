#pragma once

#include "core/Config.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "physics/EOS.hpp"

namespace blast {

// Computes L(U) = -div F  (inviscid, ideal-gas). Writes into Rhs.
// Ghost cells of U must be populated by caller (apply_bcs before each call).
// Uses 5 thread-local flux fields internally to keep memory bounded.
void compute_rhs_inviscid(const State& U, const Grid& g, const IdealGas& eos,
                          State& Rhs);

// Returns the global maximum stable timestep for the hyperbolic CFL.
Real max_dt_hyperbolic(const State& U, const Grid& g, const IdealGas& eos,
                       Real cfl);

}  // namespace blast
