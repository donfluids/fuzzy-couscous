#pragma once

#include "core/Config.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "numerics/RhsScratch.hpp"
#include "physics/EOS.hpp"
#include "physics/MixtureEOS.hpp"
#include "physics/Multifluid.hpp"
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

    // Max effective LAD (artificial diffusivity) kinematic diffusivity seen on
    // the most recent step, for the viscous CFL constraint (one-step lag).
    // Zero until the first artificial-diffusivity RHS evaluation.
    Real last_abv_nu_max() const { return scratch_.abv_nu_max; }

    // Inviscid step. Equivalent to step(U, g, bc, eos, {mu=0}, dt).
    void step(State& U, const Grid& g, const BCSet& bc, const IdealGas& eos,
              Real dt);

    // Full Navier-Stokes step: inviscid + viscous if vp.mu > 0.
    // gfn (optional): per-cell multifluid marker (held fixed across the 3
    // stages; advected separately by the caller).
    // mix (optional): marker-selected mixture EOS (two-gamma or air+JWL);
    // nullptr keeps the original gloc = 1 + 1/G arithmetic.
    void step(State& U, const Grid& g, const BCSet& bc, const IdealGas& eos,
              const ViscousParams& vp, Real dt, const Field3D* gfn = nullptr,
              const MixtureEOS* mix = nullptr);

    // Navier-Stokes step with a user-supplied source-term callback (for
    // MMS verification). `t_current` is the time at the start of the step.
    void step_with_source(State& U, const Grid& g, const BCSet& bc,
                          const IdealGas& eos, const ViscousParams& vp,
                          Real dt, Real t_current, const SourceCallback& src);

    // True five-equation step: advances the conserved State and the aux bundle
    // (Z1, Z2, alpha1) in lockstep through all 3 SSP-RK3 stages, so the
    // partial-mass fluxes see the same stage velocity as the mixture flux.
    // Applies positivity + volume-fraction boundedness after each stage.
    void step_5eq(State& U, FiveEqAux& aux, const Grid& g, const BCSet& bc,
                  const IdealGas& eos, const ViscousParams& vp, Real dt,
                  const MixtureEOS& mix);

#ifdef BLAST_MPI
    // MPI variant: halo-exchange before applying physical BCs at each stage.
    // The Halo object must outlive this RK3.
    // gfn (optional): two-gamma multifluid G field, held fixed across the 3
    // stages (its ghosts must be valid on entry; advected separately by caller).
    void step_mpi(State& U, const Grid& g, const BCSet& bc, const IdealGas& eos,
                  const ViscousParams& vp, Real dt,
                  const Domain& d, Halo& halo, const Field3D* gfn = nullptr,
                  const MixtureEOS* mix = nullptr);

    // Build the MPI pseudospectral hyperdissipation operator with the
    // global grid, Cartesian domain, and BC-driven basis mode. Call once
    // before the time loop when ViscousParams::hyper_method ==
    // Pseudospectral on an MPI run.
    void init_spectral_hyper_mpi(const Grid& global_grid, const Domain& d,
                                 SpectralBCMode mode);
#endif

private:
    State       U1_;
    State       k_;
    RhsScratch  scratch_;
    FiveEqAux   aux1_;             // five-equation: stage state (lazy-allocated)
    FiveEqAux   kaux_;             // five-equation: aux RHS accumulator
    bool        aux_allocated_ = false;
};

}  // namespace blast
