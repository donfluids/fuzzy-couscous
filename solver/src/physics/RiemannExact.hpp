#pragma once

#include "core/Types.hpp"
#include "physics/EOS.hpp"

namespace blast::riemann {

struct LR { Real rho, u, p; };
struct StarState { Real p_star, u_star; };

// Newton iteration for the star-region pressure of an exact ideal-gas Riemann
// problem (Toro Ch. 4). Robust for the canonical Sod / Lax / Shu-Osher
// initial conditions.
StarState star_state(LR L, LR R, const GammaLaw& g, int max_iter = 60,
                     Real tol = 1e-10);

// Sample the exact self-similar Riemann solution at speed s = x/t.
LR sample(LR Lc, LR Rc, const StarState& star, Real s, const GammaLaw& g);

}  // namespace blast::riemann
