#pragma once

#include "core/Config.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "physics/EOS.hpp"
#include "physics/ViscousFlux.hpp"

#ifdef BLAST_MPI
#include <mpi.h>
#endif

namespace blast {

// Computes L(U) = -div F  (inviscid, ideal-gas). Writes into Rhs.
// Ghost cells of U must be populated by caller (apply_bcs before each call).
// Uses 5 thread-local flux fields internally to keep memory bounded.
void compute_rhs_inviscid(const State& U, const Grid& g, const IdealGas& eos,
                          State& Rhs);

// Adds viscous contribution + div(tau u - q) to existing Rhs in-place.
// Caller has called apply_bcs and (if needed) compute_rhs_inviscid first.
void add_rhs_viscous(const State& U, const Grid& g, const IdealGas& eos,
                     const ViscousParams& vp, State& Rhs);

// Returns the global maximum stable timestep for the hyperbolic CFL.
Real max_dt_hyperbolic(const State& U, const Grid& g, const IdealGas& eos,
                       Real cfl);

// Returns the global maximum stable timestep for the viscous CFL,
// dt = cfl * dx^2 / nu where nu = mu/rho_min.
Real max_dt_viscous(const State& U, const Grid& g, const ViscousParams& vp,
                    Real cfl);

#ifdef BLAST_MPI
// MPI-aware dt: compute the local minimum, then MPI_Allreduce(..., MIN, comm)
// to get the global one. Same numerics as the serial version.
Real max_dt_hyperbolic(const State& U, const Grid& g, const IdealGas& eos,
                       Real cfl, MPI_Comm comm);
Real max_dt_viscous(const State& U, const Grid& g, const ViscousParams& vp,
                    Real cfl, MPI_Comm comm);
#endif

}  // namespace blast
