#include "numerics/RK3.hpp"

#include "bc/BC.hpp"
#include "numerics/RHS.hpp"

namespace blast {

RK3::RK3(int nx, int ny, int nz, int ng)
    : U1_(nx, ny, nz, ng), k_(nx, ny, nz, ng) {}

void RK3::step(State& U, const Grid& g, const BCSet& bc, const IdealGas& eos,
               Real dt) {
    // Stage 1: U1 = U^n + dt L(U^n)
    apply_bcs(U, bc);
    compute_rhs_inviscid(U, g, eos, k_);
    state_axpby(U1_, 1.0, U, dt, k_);

    // Stage 2: U2 = 3/4 U^n + 1/4 U1 + 1/4 dt L(U1).
    // We reuse U1_ for U2 by computing in place: U1_ = 3/4 U + 1/4 U1_ + 1/4 dt k.
    apply_bcs(U1_, bc);
    compute_rhs_inviscid(U1_, g, eos, k_);
    state_axpbypcz(U1_, 3.0 / 4.0, U, 1.0 / 4.0, U1_, dt / 4.0, k_);

    // Stage 3: U^{n+1} = 1/3 U^n + 2/3 U2 + 2/3 dt L(U2).
    apply_bcs(U1_, bc);
    compute_rhs_inviscid(U1_, g, eos, k_);
    state_axpbypcz(U, 1.0 / 3.0, U, 2.0 / 3.0, U1_, 2.0 / 3.0 * dt, k_);
}

}  // namespace blast
