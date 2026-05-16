#pragma once

#include "core/Config.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "physics/EOS.hpp"
#include "physics/ViscousFlux.hpp"

namespace blast {

// Storage and driver for a SSP-RK3 (Gottlieb-Shu) step.
class RK3 {
public:
    RK3(int nx, int ny, int nz, int ng);

    // Inviscid step. Equivalent to step(U, g, bc, eos, {mu=0}, dt).
    void step(State& U, const Grid& g, const BCSet& bc, const IdealGas& eos,
              Real dt);

    // Full Navier-Stokes step: inviscid + viscous if vp.mu > 0.
    // U is advanced in place; apply_bcs is called before each RHS eval.
    void step(State& U, const Grid& g, const BCSet& bc, const IdealGas& eos,
              const ViscousParams& vp, Real dt);

private:
    State U1_;
    State k_;
};

}  // namespace blast
