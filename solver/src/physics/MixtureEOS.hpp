#pragma once

#include "core/Types.hpp"
#include "physics/JWL.hpp"

#include <cmath>

namespace blast {

// Marker-selected mixture EOS. A per-cell advected scalar (the multifluid marker
// field `gfn`) selects which equation of state evaluates pressure and sound
// speed in the flux/CFL loops. Two modes:
//
//  - TwoGamma: the marker IS G = 1/(gamma-1); local gamma = 1 + 1/G. Reproduces
//    the original two-gamma multifluid arithmetic BIT-FOR-BIT (the two helper
//    forms below match the two pre-existing call-site spellings exactly), so
//    enabling MixtureEOS on a two-gamma run is a verified no-op.
//
//  - JWL: the marker is a mass fraction phi in [0,1] (phi=1 pure products,
//    phi=0 pure air). A cell is evaluated with the JWL products EOS when
//    phi >= phi_switch, else as ideal air. This "frozen one-EOS-per-cell" rule
//    matches the existing frozen-gamma contact treatment; the gated double-flux
//    + WENO contact sensor keep the interface 1-2 cells thin.
//
// This is a small free-function dispatcher (not a class hierarchy) so the
// per-cell branch in the hot flux loop stays predictable and SIMD-friendly,
// exactly as the existing `gfn ? ... : eos.gamma` branch already is.
enum class MixMode { TwoGamma, JWL };

struct MixtureEOS {
    MixMode   mode       = MixMode::TwoGamma;
    Real      gamma_air  = 1.4;     // ideal-gas gamma for the air side (JWL mode)
    Real      phi_switch = 0.5;     // products if marker >= phi_switch (JWL mode)
    JWLParams jwl{};                // products EOS (JWL mode)

    // True when this cell's marker selects the JWL products EOS.
    inline bool is_products(Real marker) const {
        return mode == MixMode::JWL && marker >= phi_switch;
    }

    // Pressure + sound speed for the "gloc" call sites (fill_flux_and_alpha,
    // max_dt_hyperbolic). TwoGamma reproduces gloc = 1 + 1/marker exactly.
    inline void p_c(Real marker, Real rho, Real e_int, Real& p, Real& c) const {
        if (is_products(marker)) {
            p = jwl_pressure(jwl, rho, e_int);
            c = std::sqrt(std::max(jwl_sound_speed2(jwl, rho, p), 0.0));
        } else {
            const Real gloc = (mode == MixMode::JWL)
                                  ? gamma_air                 // ideal air
                                  : (1.0 + 1.0 / marker);     // two-gamma (exact)
            p = (gloc - 1.0) * e_int;
            c = std::sqrt(gloc * p / rho);
        }
    }

    // Frozen pressure + sound speed for the double-flux stencil. TwoGamma
    // reproduces the pre-existing spelling gm1 = 1/marker; p = gm1*e_int;
    // c = sqrt(max(gam*p/rho, 0)) bit-for-bit. The frozen choice (products vs
    // air, set by `prod` = is_products(marker_c) of the UPDATE cell) is applied
    // to every cell in the stencil; for JWL the params are global so "frozen"
    // means "use the update cell's EOS for all 7 cells".
    inline void p_c_frozen(bool prod, Real gm1, Real gam, Real rho, Real e_int,
                           Real& p, Real& c) const {
        if (prod) {
            p = jwl_pressure(jwl, rho, e_int);
            c = std::sqrt(std::max(jwl_sound_speed2(jwl, rho, p), 0.0));
        } else {
            p = gm1 * e_int;
            c = std::sqrt(std::max(gam * p / rho, 0.0));
        }
    }

    // Frozen gamma-1 for the update cell's NON-products (two-gamma or air) path.
    // TwoGamma: 1/marker (exact). JWL air: gamma_air - 1.
    inline Real frozen_gm1(Real marker_c) const {
        return (mode == MixMode::JWL) ? (gamma_air - 1.0) : (1.0 / marker_c);
    }

    // Sound speed from a known pressure (used by the shock sensor). TwoGamma
    // reproduces sqrt((1+1/marker) p / rho).
    inline Real sound_speed(Real marker, Real rho, Real p) const {
        if (is_products(marker))
            return std::sqrt(std::max(jwl_sound_speed2(jwl, rho, p), 0.0));
        const Real gloc = (mode == MixMode::JWL) ? gamma_air : (1.0 + 1.0 / marker);
        return std::sqrt(gloc * p / rho);
    }
};

}  // namespace blast
