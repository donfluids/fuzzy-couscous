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

struct RhsScratch;  // numerics/RhsScratch.hpp

// Computes L(U) = -div F  (inviscid, ideal-gas). Writes into Rhs.
// Ghost cells of U must be populated by caller (apply_bcs before each call).
//
// Two overloads: the bare one allocates local scratch on the stack (used by
// tests and ad-hoc calls); the scratch-aware one uses pre-allocated buffers
// from RhsScratch, which is how RK3 wires it in production (avoids ~1.5 GB
// of allocator + first-touch traffic per step at 256^3).
// gfn (optional): per-cell G = 1/(gamma-1) field for a two-gamma multifluid.
// nullptr -> single ideal gas (eos.gamma everywhere; existing behavior).
void compute_rhs_inviscid(const State& U, const Grid& g, const IdealGas& eos,
                          State& Rhs, const Field3D* gfn = nullptr);
void compute_rhs_inviscid(const State& U, const Grid& g, const IdealGas& eos,
                          RhsScratch& scratch, State& Rhs,
                          const Field3D* gfn = nullptr);

// Adds viscous contribution + div(tau u - q) to existing Rhs in-place.
// Caller has called apply_bcs and (if needed) compute_rhs_inviscid first.
void add_rhs_viscous(const State& U, const Grid& g, const IdealGas& eos,
                     const ViscousParams& vp, State& Rhs);
void add_rhs_viscous(const State& U, const Grid& g, const IdealGas& eos,
                     const ViscousParams& vp, RhsScratch& scratch, State& Rhs);

// Returns the global maximum stable timestep for the hyperbolic CFL.
Real max_dt_hyperbolic(const State& U, const Grid& g, const IdealGas& eos,
                       Real cfl, const Field3D* gfn = nullptr);

// Returns the global maximum stable timestep for the viscous CFL,
// dt = cfl * dx^2 / nu where nu = mu/rho_min.
Real max_dt_viscous(const State& U, const Grid& g, const ViscousParams& vp,
                    Real cfl);

#ifdef BLAST_MPI
// MPI-aware dt: compute the local minimum, then MPI_Allreduce(..., MIN, comm)
// to get the global one. Same numerics as the serial version.
Real max_dt_hyperbolic(const State& U, const Grid& g, const IdealGas& eos,
                       Real cfl, MPI_Comm comm, const Field3D* gfn = nullptr);
Real max_dt_viscous(const State& U, const Grid& g, const ViscousParams& vp,
                    Real cfl, MPI_Comm comm);
#endif

}  // namespace blast
