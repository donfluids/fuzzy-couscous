#pragma once

#include "core/Types.hpp"

#include <cmath>

namespace blast {

struct Prim {
    Real rho, u, v, w, p, T;
};

// Ideal-gas conversions. e is *specific* internal energy = e_int / rho.
struct IdealGas {
    GammaLaw eos;

    Real pressure(Real rho, Real e_int) const {
        return (eos.gamma - 1.0) * e_int;
    }
    Real temperature(Real rho, Real p) const {
        return p / (rho * eos.R);
    }
    Real sound_speed(Real rho, Real p) const {
        return std::sqrt(eos.gamma * p / rho);
    }

    // (rho, rho u, rho v, rho w, rho E) -> primitive.
    Prim to_prim(Real rho, Real mx, Real my, Real mz, Real rhoE) const {
        Prim P;
        P.rho = rho;
        const Real inv_rho = 1.0 / rho;
        P.u = mx * inv_rho;
        P.v = my * inv_rho;
        P.w = mz * inv_rho;
        const Real ke = 0.5 * rho * (P.u*P.u + P.v*P.v + P.w*P.w);
        const Real e_int = rhoE - ke;     // total internal energy density
        P.p = pressure(rho, e_int);
        P.T = temperature(rho, P.p);
        return P;
    }
};

}  // namespace blast
