#pragma once

#include "core/Types.hpp"

#include <cmath>

namespace blast {

// Jones-Wilkins-Lee (JWL) EOS for detonation products (e.g. TNT). Real high
// explosives do NOT obey an ideal-gas law: the products pressure is set by a
// pair of exponential reference terms (the cold/isentrope curve) plus a
// Grueneisen thermal term. We use the Mie-Grueneisen form
//
//   p(rho, e) = A (1 - w/(R1 V)) exp(-R1 V)
//             + B (1 - w/(R2 V)) exp(-R2 V)
//             + w * e_int
//
// where V = rho0/rho is the relative volume (rho0 = solid/reference density),
// e_int is the internal-energy DENSITY (per unit current volume, = rhoE - ke,
// the same quantity the solver carries), and w = omega is the Grueneisen
// coefficient. The cubic "C V^-(w+1)" isentrope term cancels out of the
// p(rho,e) form, so only {A, B, R1, R2, omega, rho0} are needed to evaluate the
// EOS (the detonation energy E0 only sets the products' initial e_int).
//
// Reduction check: with A=B=0 (no reference curve) p = w*e_int and c^2 =
// (1+w) p / rho -- i.e. an ideal gas with gamma = 1 + omega. This is the hook
// that lets the mixture EOS treat air (ideal) and products (JWL) uniformly.
//
// Units: A, B, rho0 carry dimensions (pressure, pressure, density); R1, R2,
// omega are dimensionless. The formula is homogeneous, so a nondimensional run
// simply stores A/p_ref, B/p_ref, rho0/rho_ref (done once at config load).
struct JWLParams {
    Real A = 0.0, B = 0.0;        // reference-curve coefficients [pressure]
    Real R1 = 0.0, R2 = 0.0;      // reference-curve exponents [-]
    Real omega = 0.0;             // Grueneisen coefficient [-]
    Real rho0 = 1.0;              // reference (solid) density
    Real E0 = 0.0;                // detonation energy per unit volume [pressure]
};

// Reference-curve part f(V) = A(1-w/(R1 V))e^{-R1 V} + B(1-w/(R2 V))e^{-R2 V}.
inline Real jwl_fref(const JWLParams& j, Real V) {
    const Real e1 = std::exp(-j.R1 * V);
    const Real e2 = std::exp(-j.R2 * V);
    return j.A * (1.0 - j.omega / (j.R1 * V)) * e1
         + j.B * (1.0 - j.omega / (j.R2 * V)) * e2;
}

// p(rho, e_int): e_int is the internal-energy density (rhoE - ke).
inline Real jwl_pressure(const JWLParams& j, Real rho, Real e_int) {
    const Real V = j.rho0 / rho;
    return jwl_fref(j, V) + j.omega * e_int;
}

// Invert at fixed rho: e_int such that jwl_pressure(rho, e_int) = p. Closed form
// (the thermal term is linear in e_int). Used by the IC to set rhoE from a
// target (rho, p) products state.
inline Real jwl_eint_from_p(const JWLParams& j, Real rho, Real p) {
    const Real V = j.rho0 / rho;
    return (p - jwl_fref(j, V)) / j.omega;
}

// Isentropic sound speed squared: c^2 = (dp/drho)|_s. With p = f(V) + w*e_int
// and the isentrope relation de = (p/rho^2) drho, this works out to
//
//   c^2 = [ (1+w) p - f(V) - V f'(V) ] / rho.
//
// (Reduces to gamma p/rho with gamma=1+w when f=0; verified against TNT's CJ
// point: rho0=1630, p_CJ=21 GPa, D=6930 -> c = 5070 m/s = D - u_CJ.)
inline Real jwl_sound_speed2(const JWLParams& j, Real rho, Real p) {
    const Real V  = j.rho0 / rho;
    const Real e1 = std::exp(-j.R1 * V);
    const Real e2 = std::exp(-j.R2 * V);
    const Real f  = j.A * (1.0 - j.omega / (j.R1 * V)) * e1
                  + j.B * (1.0 - j.omega / (j.R2 * V)) * e2;
    // f'(V) = A e1 (w/(R1 V^2) - R1 + w/V) + B e2 (w/(R2 V^2) - R2 + w/V)
    const Real fp = j.A * e1 * (j.omega / (j.R1 * V * V) - j.R1 + j.omega / V)
                  + j.B * e2 * (j.omega / (j.R2 * V * V) - j.R2 + j.omega / V);
    return ((1.0 + j.omega) * p - f - V * fp) / rho;
}

}  // namespace blast
