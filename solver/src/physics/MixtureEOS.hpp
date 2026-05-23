#pragma once

#include "core/Types.hpp"
#include "physics/JWL.hpp"
#include "physics/StiffenedGas.hpp"

#include <algorithm>
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
// FiveEquation mode: a true Allaire-Kapila-Massoni (2002) diffuse-interface
// model. Two phases coexist in every cell with volume fractions (alpha1, 1-alpha1)
// and partial densities (Z1 = alpha1*rho1, Z2 = alpha2*rho2). Each phase has its
// own EOS (stiffened gas OR JWL); pressure and the frozen sound speed come from
// the volume-fraction-averaged mixture law below. This is selected via
// MixMode::FiveEquation and the `phase[2]` array; the TwoGamma/JWL members and
// methods above are untouched so those modes stay bit-for-bit identical.
enum class MixMode { TwoGamma, JWL, FiveEquation };

// One phase of the five-equation mixture. Either a stiffened gas (ideal gas is
// the pinf=0 case) or a JWL EOS. The unified Grueneisen split rho_k e_k =
// (p + Phi_k)/(Gamma_k - 1) drives the mixture law:
//   Gamma_k - 1 = gamma_k - 1 (SG) or omega_k (JWL)
//   Phi_k       = gamma_k pinf_k (SG) or -f_ref(V_k) (JWL), V_k = rho0_k/rho_k.
struct PhaseEOS {
    enum Kind { StiffenedGas, JWLPhase };
    Kind               kind = StiffenedGas;
    StiffenedGasParams sg{};
    JWLParams          jwl{};

    // Effective (Gamma - 1) of the Grueneisen split.
    inline Real gm1() const {
        return kind == StiffenedGas ? (sg.gamma - 1.0) : jwl.omega;
    }
    // Reference term Phi_k in rho_k e_k = (p + Phi_k)/(Gamma_k - 1).
    inline Real phi(Real rho_k) const {
        return kind == StiffenedGas ? (sg.gamma * sg.pinf)
                                    : -jwl_fref(jwl, jwl.rho0 / rho_k);
    }
    // Single-phase isentropic sound speed squared at (rho_k, p).
    inline Real c2(Real rho_k, Real p) const {
        return kind == StiffenedGas ? sg_sound_speed2(sg, rho_k, p)
                                    : jwl_sound_speed2(jwl, rho_k, p);
    }
};

// Frozen mixture coefficients of the update cell, applied across the double-flux
// stencil (the five-equation analogue of the frozen gamma). The mixture behaves
// locally like a single stiffened gas with (Gamma, Pinf), so a uniform-(p,u)
// interface stays oscillation-free.
struct FiveEqFrozen {
    Real S;       // sum_k alpha_k/(Gamma_k - 1)
    Real Pi;      // sum_k alpha_k Phi_k/(Gamma_k - 1)
    Real Gamma;   // 1 + 1/S
    Real Pinf;    // Pi/(S + 1)
};

struct MixtureEOS {
    MixMode   mode       = MixMode::TwoGamma;
    Real      gamma_air  = 1.4;     // ideal-gas gamma for the air side (JWL mode)
    Real      phi_switch = 0.5;     // products if marker >= phi_switch (JWL mode)
    JWLParams jwl{};                // products EOS (JWL mode)

    PhaseEOS  phase[2];             // FiveEquation: phase 0 and phase 1
    Real      a_floor = 1e-6;       // volume-fraction floor (keep both phases)

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

    // ---- FiveEquation mixture law ----------------------------------------
    // Mixture "1/(Gamma-1)" (S) and reference sum (Pi) plus the phase densities,
    // from the volume fraction and partial densities. alpha clamped to keep both
    // phases present (avoids rho_k = Z_k/alpha_k blowing up).
    inline void mix_SPi(Real a1, Real Z1, Real Z2, Real& S, Real& Pi,
                        Real& rho1, Real& rho2) const {
        const Real a1c = std::min(std::max(a1, a_floor), 1.0 - a_floor);
        const Real a2c = 1.0 - a1c;
        rho1 = Z1 / a1c;
        rho2 = Z2 / a2c;
        const Real g1 = phase[0].gm1(), g2 = phase[1].gm1();
        S  = a1c / g1 + a2c / g2;
        Pi = a1c * phase[0].phi(rho1) / g1 + a2c * phase[1].phi(rho2) / g2;
    }

    // Pressure (volume-fraction-averaged) + frozen mass-weighted mixture sound
    // speed c^2 = sum_k Y_k c_k^2 (the Allaire-model characteristic speed).
    inline void p_c_5eq(Real a1, Real Z1, Real Z2, Real rho, Real rhoe,
                        Real& p, Real& c) const {
        Real S, Pi, rho1, rho2;
        mix_SPi(a1, Z1, Z2, S, Pi, rho1, rho2);
        p = (rhoe - Pi) / S;
        const Real Y1 = Z1 / rho, Y2 = Z2 / rho;
        const Real c2m = Y1 * std::max(phase[0].c2(rho1, p), 0.0)
                       + Y2 * std::max(phase[1].c2(rho2, p), 0.0);
        c = std::sqrt(std::max(c2m, 0.0));
    }

    // Internal-energy density rhoe that yields pressure p at this composition
    // (inverse of p_c_5eq's pressure). Used by the IC so the flux-loop pressure
    // matches the initial state exactly.
    inline Real five_eq_rhoe_from_p(Real a1, Real Z1, Real Z2, Real p) const {
        Real S, Pi, rho1, rho2;
        mix_SPi(a1, Z1, Z2, S, Pi, rho1, rho2);
        return p * S + Pi;
    }

    // Freeze the update cell's mixture coefficients for the double-flux stencil.
    inline FiveEqFrozen five_eq_freeze(Real a1, Real Z1, Real Z2) const {
        Real S, Pi, rho1, rho2;
        mix_SPi(a1, Z1, Z2, S, Pi, rho1, rho2);
        FiveEqFrozen f;
        f.S = S;
        f.Pi = Pi;
        f.Gamma = 1.0 + 1.0 / S;
        f.Pinf = Pi / (S + 1.0);
        return f;
    }

    // Frozen pressure + sound speed for a stencil cell (local rho, rhoe) using
    // the update cell's frozen coefficients: the mixture acts as a single
    // stiffened gas (Gamma, Pinf), so a uniform-(p,u) contact stays uniform.
    inline void p_c_5eq_frozen(const FiveEqFrozen& f, Real rho, Real rhoe,
                               Real& p, Real& c) const {
        p = (rhoe - f.Pi) / f.S;
        c = std::sqrt(std::max(f.Gamma * (p + f.Pinf) / rho, 0.0));
    }
};

}  // namespace blast
