#pragma once

#include "core/Types.hpp"
#include "physics/EOS.hpp"

namespace blast {

// Selects the discretization used for the hyperdissipation operator.
// FiniteDifference is the default (composed Laplacian stencils, works for
// any BC). Pseudospectral evaluates the operator exactly in spectral space
// via FFT round-trip; requires uniformly periodic or uniformly slip-wall
// BCs and a single MPI rank.
enum class HyperMethod { FiniteDifference, Pseudospectral };

// BC-driven basis for the pseudospectral path. Periodic uses complex DFT
// (r2c + c2r); SlipWall uses real-to-real DCT-II / DST-II per axis with
// per-variable kind selection (wall-normal momentum is odd, everything
// else is even). Set automatically in to_viscous() from BCSet; users do
// not configure this directly.
enum class SpectralBCMode { Periodic, SlipWall };

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
    // Higher-power hyperdissipation:  d U / d t  +=  +nu_h6 * (nabla^2)^3 U
    // (note: sign flips relative to nabla^4; (-1)^(n+1) for the 2n-th
    // power Laplacian). Fourier weight nu_h6 k^6 is even more selective
    // for the grid-scale modes than k^4, leaving the resolved range
    // less damped. Computed via three composed 4th-order accurate
    // Laplacians (radius 2 each) so it still fits NGHOST = 6. <=0 disables.
    Real hyper6_coeff = 0.0;
    // Discretization for both hyperdissipation orders above. Pseudospectral
    // is exact-on-the-grid (no FD truncation error in the operator) but
    // requires uniformly periodic or uniformly slip-wall BCs and is
    // serial-only.
    HyperMethod hyper_method = HyperMethod::FiniteDifference;
    // Selects the spectral basis when hyper_method == Pseudospectral.
    // Ignored otherwise.
    SpectralBCMode spectral_bc_mode = SpectralBCMode::Periodic;
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
