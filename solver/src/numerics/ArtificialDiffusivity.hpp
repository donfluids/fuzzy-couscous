#pragma once

#include "core/Field3D.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "numerics/Gradients.hpp"
#include "physics/EOS.hpp"
#include "physics/ViscousFlux.hpp"

namespace blast {

// Localized artificial diffusivity (LAD, Kawai & Lele 2008) fields.
//
// Each artificial transport coefficient is large only where a high-derivative
// sensor of the relevant quantity is sharp (i.e. at shocks / contacts) and
// vanishes (as a power of the grid spacing) in smooth flow:
//
//   beta_art  = C_beta  * rho * H(-div u) * |D^2(div u)| * h^2     (bulk; shocks)
//   mu_art    = C_mu    * rho *            |D^2 |S||      * h^2     (shear)
//   kappa_art = C_kappa * (rho c / T) *    |D^2 e|        * h       (thermal; entropy)
//   d_art     = C_D     * c * (|D^2 rho| / rho)           * h       (mass; contacts)
//
// d_art smooths density contacts (rho jump at ~constant u,p) that the
// dilatation/strain/energy sensors miss; it is applied as a consistent mass
// diffusion (mass + u-weighted momentum + KE-weighted energy) so u and p are
// preserved across the contact.
//
// where |S| is the strain-rate magnitude, e the specific internal energy,
// c the sound speed, h the representative cell size, and D^2 the (grid-scaled)
// 2nd difference. The 2nd-derivative sensor (vs. the canonical 4th) keeps the
// stencil within NGHOST=6; the artificial fluxes are differenced with a compact
// 2nd-order divergence (see RHS.cpp), which is all this sensor width supports.
//
// theta_src / strain_src are working buffers (filled on [-3, n+3)). The output
// fields mu_art / beta_art / kappa_art are filled on [-1, n+1) -- exactly what
// the compact divergence on the interior [0, n) needs. primT must be valid on
// [-3, n+3) (compute_cell_gradients fills it on the full ghost region).
//
// Returns the maximum effective kinematic diffusivity over the grid
//   max( (mu_art + |beta_art|)/rho ,  kappa_art/(rho c_p) )
// for the viscous CFL constraint.
Real compute_lad_fields(const State& U, const Grid& g, const IdealGas& eos,
                        const ViscousParams& vp, const CellGradients& Grad,
                        const Field3D& primT,
                        Field3D& theta_src, Field3D& strain_src,
                        Field3D& mu_art, Field3D& beta_art, Field3D& kappa_art,
                        Field3D& d_art);

}  // namespace blast
