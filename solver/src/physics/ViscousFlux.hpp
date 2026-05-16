#pragma once

#include "core/Types.hpp"
#include "physics/EOS.hpp"

namespace blast {

struct ViscousParams {
    Real mu = 1.8e-5;       // dynamic viscosity (constant for now)
    Real prandtl = 0.71;
    Real bulk_visc = 0.0;   // optional second viscosity coefficient
    // Hyperdissipation coefficient nu_h for the operator
    //   d U / d t  +=  -nu_h * (nabla^2)^2 U
    // applied uniformly to every conserved variable. <=0 disables.
    // Acts like a Smagorinsky-style high-k sink without the Smagorinsky
    // model's near-wall pathology: linear in U so it's cheap, and only
    // attenuates modes near the Nyquist (Fourier weight nu_h k^4 grows
    // fast). Standard LES stability device when central / WENO schemes
    // alone produce zero numerical dissipation in smooth turbulence.
    Real hyper_coeff = 0.0;
};

// Local 3x3 velocity gradient at a cell, dudx[v][d] = d u_v / d x_d,
// plus the temperature gradient and velocity components.
struct CellState {
    Real u, v, w, T;
    Real dudx[3][3];
    Real dTdx[3];
};

// Viscous flux vector G_d for direction d (Stokes form, constant mu).
//   G_d[RHO ]  = 0
//   G_d[RHOU] = tau_d,0
//   G_d[RHOV] = tau_d,1
//   G_d[RHOW] = tau_d,2
//   G_d[RHOE] = u_i tau_d,i  -  q_d
struct ViscousFluxVec {
    Real f[NCONS];
};

inline ViscousFluxVec viscous_flux(const CellState& C,
                                   const ViscousParams& vp,
                                   const IdealGas& eos, int d) {
    const Real div_u = C.dudx[0][0] + C.dudx[1][1] + C.dudx[2][2];

    // Stress tensor row d: tau_d,j = mu*(du_d/dx_j + du_j/dx_d) + (beta - 2mu/3) delta_dj div_u
    Real tau[3];
    const Real lambda = vp.bulk_visc - (2.0 / 3.0) * vp.mu;
    for (int j = 0; j < 3; ++j) {
        tau[j] = vp.mu * (C.dudx[d][j] + C.dudx[j][d]);
        if (j == d) tau[j] += lambda * div_u;
    }

    // Conductivity from constant Prandtl.
    const Real kappa = vp.mu * eos.eos.cp() / vp.prandtl;
    const Real q_d   = -kappa * C.dTdx[d];

    ViscousFluxVec G{};
    G.f[RHO ] = 0.0;
    G.f[RHOU] = tau[0];
    G.f[RHOV] = tau[1];
    G.f[RHOW] = tau[2];
    G.f[RHOE] = C.u * tau[0] + C.v * tau[1] + C.w * tau[2] - q_d;
    return G;
}

}  // namespace blast
