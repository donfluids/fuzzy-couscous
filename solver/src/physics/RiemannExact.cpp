#include "physics/RiemannExact.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace blast::riemann {

namespace {

Real sound_speed(const LR& s, const GammaLaw& g) {
    return std::sqrt(g.gamma * s.p / s.rho);
}

// Toro eq. 4.6 for pressure function f_K and its derivative df_K/dp.
void fK(Real p, const LR& K, const GammaLaw& g, Real& f, Real& fp) {
    const Real cK = sound_speed(K, g);
    if (p > K.p) {
        // Shock branch.
        const Real A = 2.0 / ((g.gamma + 1.0) * K.rho);
        const Real B = (g.gamma - 1.0) / (g.gamma + 1.0) * K.p;
        const Real q = std::sqrt(A / (p + B));
        f  = (p - K.p) * q;
        fp = q * (1.0 - 0.5 * (p - K.p) / (B + p));
    } else {
        // Rarefaction branch.
        const Real pr = p / K.p;
        f  = 2.0 * cK / (g.gamma - 1.0) * (std::pow(pr, (g.gamma - 1.0) / (2.0 * g.gamma)) - 1.0);
        fp = std::pow(pr, -(g.gamma + 1.0) / (2.0 * g.gamma)) / (K.rho * cK);
    }
}

}  // namespace

StarState star_state(LR L, LR R, const GammaLaw& g, int max_iter, Real tol) {
    // Initial guess: two-rarefaction approximation.
    const Real cL = sound_speed(L, g);
    const Real cR = sound_speed(R, g);
    const Real gm1_2g = (g.gamma - 1.0) / (2.0 * g.gamma);
    const Real twogm1 = 2.0 * g.gamma / (g.gamma - 1.0);

    Real p0;
    {
        Real num = cL + cR - 0.5 * (g.gamma - 1.0) * (R.u - L.u);
        Real den = cL / std::pow(L.p, gm1_2g) + cR / std::pow(R.p, gm1_2g);
        p0 = std::pow(num / den, twogm1);
        p0 = std::max(p0, 1e-12);
    }

    Real p = p0;
    for (int it = 0; it < max_iter; ++it) {
        Real fL, fpL, fR, fpR;
        fK(p, L, g, fL, fpL);
        fK(p, R, g, fR, fpR);
        const Real f  = fL + fR + (R.u - L.u);
        const Real fp = fpL + fpR;
        const Real dp = f / fp;
        Real p_new = p - dp;
        if (p_new < 1e-14) p_new = 0.5 * p;  // safeguard
        if (std::fabs(p_new - p) < tol * (p + p_new + 1e-30)) {
            p = p_new;
            break;
        }
        p = p_new;
    }

    Real fL, fpL, fR, fpR;
    fK(p, L, g, fL, fpL);
    fK(p, R, g, fR, fpR);
    const Real u_star = 0.5 * (L.u + R.u) + 0.5 * (fR - fL);
    return {p, u_star};
}

LR sample(LR Lc, LR Rc, const StarState& star, Real s, const GammaLaw& g) {
    const Real cL = sound_speed(Lc, g);
    const Real cR = sound_speed(Rc, g);
    const Real gm1 = g.gamma - 1.0;
    const Real gp1 = g.gamma + 1.0;
    const Real pratL = star.p_star / Lc.p;
    const Real pratR = star.p_star / Rc.p;

    if (s <= star.u_star) {
        // Left of contact.
        if (star.p_star <= Lc.p) {
            // Left rarefaction.
            const Real SHL = Lc.u - cL;
            if (s <= SHL) return Lc;
            const Real cmL = cL * std::pow(pratL, gm1 / (2.0 * g.gamma));
            const Real STL = star.u_star - cmL;
            if (s >= STL) {
                LR out;
                out.rho = Lc.rho * std::pow(pratL, 1.0 / g.gamma);
                out.u   = star.u_star;
                out.p   = star.p_star;
                return out;
            }
            // Inside fan.
            LR out;
            out.u   = (2.0 / gp1) * (cL + 0.5 * gm1 * Lc.u + s);
            const Real c = (2.0 / gp1) * (cL + 0.5 * gm1 * (Lc.u - s));
            out.rho = Lc.rho * std::pow(c / cL, 2.0 / gm1);
            out.p   = Lc.p   * std::pow(c / cL, 2.0 * g.gamma / gm1);
            return out;
        } else {
            // Left shock.
            const Real SL = Lc.u - cL * std::sqrt((gp1 / (2.0 * g.gamma)) * pratL + gm1 / (2.0 * g.gamma));
            if (s <= SL) return Lc;
            LR out;
            out.rho = Lc.rho * (pratL + gm1 / gp1) / (gm1 / gp1 * pratL + 1.0);
            out.u   = star.u_star;
            out.p   = star.p_star;
            return out;
        }
    } else {
        // Right of contact.
        if (star.p_star <= Rc.p) {
            // Right rarefaction.
            const Real SHR = Rc.u + cR;
            if (s >= SHR) return Rc;
            const Real cmR = cR * std::pow(pratR, gm1 / (2.0 * g.gamma));
            const Real STR = star.u_star + cmR;
            if (s <= STR) {
                LR out;
                out.rho = Rc.rho * std::pow(pratR, 1.0 / g.gamma);
                out.u   = star.u_star;
                out.p   = star.p_star;
                return out;
            }
            LR out;
            out.u   = (2.0 / gp1) * (-cR + 0.5 * gm1 * Rc.u + s);
            const Real c = (2.0 / gp1) * (cR - 0.5 * gm1 * (Rc.u - s));
            out.rho = Rc.rho * std::pow(c / cR, 2.0 / gm1);
            out.p   = Rc.p   * std::pow(c / cR, 2.0 * g.gamma / gm1);
            return out;
        } else {
            const Real SR = Rc.u + cR * std::sqrt((gp1 / (2.0 * g.gamma)) * pratR + gm1 / (2.0 * g.gamma));
            if (s >= SR) return Rc;
            LR out;
            out.rho = Rc.rho * (pratR + gm1 / gp1) / (gm1 / gp1 * pratR + 1.0);
            out.u   = star.u_star;
            out.p   = star.p_star;
            return out;
        }
    }
}

}  // namespace blast::riemann
