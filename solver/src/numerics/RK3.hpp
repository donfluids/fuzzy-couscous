#pragma once

#include "core/Config.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "physics/EOS.hpp"

namespace blast {

// Storage and driver for a SSP-RK3 (Gottlieb-Shu) step.
class RK3 {
public:
    RK3(int nx, int ny, int nz, int ng);

    // U is advanced in place. apply_bcs is called before each RHS eval.
    void step(State& U, const Grid& g, const BCSet& bc, const IdealGas& eos,
              Real dt);

private:
    State U1_;
    State k_;
};

}  // namespace blast
