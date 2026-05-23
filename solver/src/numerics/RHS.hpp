#pragma once

#include "core/Config.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "physics/EOS.hpp"
#include "physics/MixtureEOS.hpp"
#include "physics/ViscousFlux.hpp"

#ifdef BLAST_MPI
#include <mpi.h>
#endif

namespace blast {

struct RhsScratch;  // numerics/RhsScratch.hpp
struct FiveEqAux;   // physics/Multifluid.hpp

// Computes L(U) = -div F  (inviscid, ideal-gas). Writes into Rhs.
// Ghost cells of U must be populated by caller (apply_bcs before each call).
//
// Two overloads: the bare one allocates local scratch on the stack (used by
// tests and ad-hoc calls); the scratch-aware one uses pre-allocated buffers
// from RhsScratch, which is how RK3 wires it in production (avoids ~1.5 GB
// of allocator + first-touch traffic per step at 256^3).
// gfn (optional): per-cell marker field for a multifluid (two-gamma G, or a JWL
// products mass fraction). nullptr -> single ideal gas (eos.gamma everywhere;
// existing behavior).
// mix (optional): marker-selected mixture EOS (TwoGamma or air+JWL). nullptr
// with a non-null gfn keeps the original two-gamma arithmetic (gloc = 1 + 1/G).
// aux5/auxRhs (optional): the five-equation aux bundle (Z1,Z2,alpha1) and its
// RHS accumulator. When mix->mode == FiveEquation and both are non-null, the
// true five-equation path runs (conservative momentum/energy + derived mixture
// mass + non-conservative volume fraction); auxRhs is overwritten.
void compute_rhs_inviscid(const State& U, const Grid& g, const IdealGas& eos,
                          State& Rhs, const Field3D* gfn = nullptr,
                          const MixtureEOS* mix = nullptr,
                          const FiveEqAux* aux5 = nullptr,
                          FiveEqAux* auxRhs = nullptr);
void compute_rhs_inviscid(const State& U, const Grid& g, const IdealGas& eos,
                          RhsScratch& scratch, State& Rhs,
                          const Field3D* gfn = nullptr,
                          const MixtureEOS* mix = nullptr,
                          const FiveEqAux* aux5 = nullptr,
                          FiveEqAux* auxRhs = nullptr);

// Adds viscous contribution + div(tau u - q) to existing Rhs in-place.
// Caller has called apply_bcs and (if needed) compute_rhs_inviscid first.
void add_rhs_viscous(const State& U, const Grid& g, const IdealGas& eos,
                     const ViscousParams& vp, State& Rhs);
void add_rhs_viscous(const State& U, const Grid& g, const IdealGas& eos,
                     const ViscousParams& vp, RhsScratch& scratch, State& Rhs);

// Returns the global maximum stable timestep for the hyperbolic CFL.
Real max_dt_hyperbolic(const State& U, const Grid& g, const IdealGas& eos,
                       Real cfl, const Field3D* gfn = nullptr,
                       const MixtureEOS* mix = nullptr,
                       const FiveEqAux* aux5 = nullptr);

// Returns the global maximum stable timestep for the viscous CFL,
// dt = cfl * dx^2 / nu where nu = mu/rho_min.
Real max_dt_viscous(const State& U, const Grid& g, const ViscousParams& vp,
                    Real cfl);

#ifdef BLAST_MPI
// MPI-aware dt: compute the local minimum, then MPI_Allreduce(..., MIN, comm)
// to get the global one. Same numerics as the serial version.
Real max_dt_hyperbolic(const State& U, const Grid& g, const IdealGas& eos,
                       Real cfl, MPI_Comm comm, const Field3D* gfn = nullptr,
                       const MixtureEOS* mix = nullptr,
                       const FiveEqAux* aux5 = nullptr);
Real max_dt_viscous(const State& U, const Grid& g, const ViscousParams& vp,
                    Real cfl, MPI_Comm comm);
#endif

}  // namespace blast
