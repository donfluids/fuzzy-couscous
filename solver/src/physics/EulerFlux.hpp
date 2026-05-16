#pragma once

#include "core/Types.hpp"
#include "physics/EOS.hpp"

namespace blast {

struct ConsCell {
    Real rho, mx, my, mz, rhoE;
};

struct FluxVec {
    Real f[NCONS];
};

// Inviscid flux of (rho, rho u_i, rho E) in direction d \in {0,1,2}.
inline FluxVec euler_flux(const ConsCell& U, Real p, int d) {
    FluxVec F{};
    const Real inv_rho = 1.0 / U.rho;
    const Real u = U.mx * inv_rho;
    const Real v = U.my * inv_rho;
    const Real w = U.mz * inv_rho;
    const Real ud = (d == 0 ? u : d == 1 ? v : w);

    F.f[RHO]  = U.rho * ud;
    F.f[RHOU] = U.mx * ud + (d == 0 ? p : 0.0);
    F.f[RHOV] = U.my * ud + (d == 1 ? p : 0.0);
    F.f[RHOW] = U.mz * ud + (d == 2 ? p : 0.0);
    F.f[RHOE] = (U.rhoE + p) * ud;
    return F;
}

}  // namespace blast
