#pragma once

#include "core/Config.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "physics/EOS.hpp"
#include "physics/ViscousFlux.hpp"

#ifdef BLAST_MPI
#include "parallel/Domain.hpp"
#include "parallel/Halo.hpp"
#endif

#include <functional>

namespace blast {

// Source-term callback. Called after compute_rhs_inviscid + add_rhs_viscous
// at each RK3 stage; should ADD (not overwrite) source contributions to the
// per-conserved-variable RHS arrays. `t_stage` is the time at which the RHS
// is being evaluated (stages of SSP-RK3 evaluate at t^n, t^n+dt, t^n+dt/2).
using SourceCallback = std::function<void(State& Rhs, const Grid& g, Real t_stage)>;

// Storage and driver for a SSP-RK3 (Gottlieb-Shu) step.
class RK3 {
public:
    RK3(int nx, int ny, int nz, int ng);

    // Inviscid step. Equivalent to step(U, g, bc, eos, {mu=0}, dt).
    void step(State& U, const Grid& g, const BCSet& bc, const IdealGas& eos,
              Real dt);

    // Full Navier-Stokes step: inviscid + viscous if vp.mu > 0.
    void step(State& U, const Grid& g, const BCSet& bc, const IdealGas& eos,
              const ViscousParams& vp, Real dt);

    // Navier-Stokes step with a user-supplied source-term callback (for
    // MMS verification). `t_current` is the time at the start of the step.
    void step_with_source(State& U, const Grid& g, const BCSet& bc,
                          const IdealGas& eos, const ViscousParams& vp,
                          Real dt, Real t_current, const SourceCallback& src);

#ifdef BLAST_MPI
    // MPI variant: halo-exchange before applying physical BCs at each stage.
    // The Halo object must outlive this RK3.
    void step_mpi(State& U, const Grid& g, const BCSet& bc, const IdealGas& eos,
                  const ViscousParams& vp, Real dt,
                  const Domain& d, Halo& halo);
#endif

private:
    State U1_;
    State k_;
};

}  // namespace blast
