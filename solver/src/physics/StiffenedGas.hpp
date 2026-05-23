#pragma once

#include "core/Types.hpp"

#include <cmath>

namespace blast {

// Stiffened-gas (Noble-Abel/Tammann) EOS for a single phase of the
// five-equation diffuse-interface model. The stiffened gas adds a constant
// "stiffness" pressure p_inf to the ideal-gas law so that (nearly) incompressible
// materials -- water, metals, condensed explosives -- can be modelled with the
// same Grueneisen-type closure as a gas:
//
//   p(rho, e) = (gamma - 1) rho e - gamma p_inf
//
// where e is the SPECIFIC internal energy and (rho e) = e_int is the
// internal-energy DENSITY the solver carries (= rhoE - ke). Note the EOS is
// independent of rho at fixed e_int (the rho argument is kept for a uniform
// per-phase EOS signature with JWL.hpp).
//
// Reduction check: with p_inf = 0 this is an ideal gas, p = (gamma-1) e_int and
// c^2 = gamma p / rho -- the hook that lets a "stiffened gas" phase double as an
// ideal-gas phase (the analogue of JWL's A=B=0 reduction).
struct StiffenedGasParams {
    Real gamma = 1.4;   // ratio of specific heats [-]
    Real pinf  = 0.0;   // stiffness pressure p_inf [pressure]; ideal gas: 0
};

// p(rho, e_int): e_int is the internal-energy density (rhoE - ke).
inline Real sg_pressure(const StiffenedGasParams& s, Real rho, Real e_int) {
    (void)rho;
    return (s.gamma - 1.0) * e_int - s.gamma * s.pinf;
}

// Invert at fixed rho: internal-energy density e_int such that p(rho,e_int)=p.
// Closed form: e_int = (p + gamma p_inf) / (gamma - 1). Used by the IC to set
// rhoE from a target (rho, p) phase state.
inline Real sg_eint_from_p(const StiffenedGasParams& s, Real rho, Real p) {
    (void)rho;
    return (p + s.gamma * s.pinf) / (s.gamma - 1.0);
}

// Isentropic sound speed squared: c^2 = gamma (p + p_inf) / rho. Reduces to
// gamma p / rho when p_inf = 0.
inline Real sg_sound_speed2(const StiffenedGasParams& s, Real rho, Real p) {
    return s.gamma * (p + s.pinf) / rho;
}

}  // namespace blast
