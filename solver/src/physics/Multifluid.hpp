#pragma once

#include "core/Config.hpp"   // BCSet
#include "core/Field3D.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"

#ifdef BLAST_MPI
#include "parallel/Domain.hpp"
#include "parallel/Halo.hpp"
#endif

namespace blast {

// Two-gamma multifluid: a per-cell field G = 1/(gamma-1) carries the local fluid
// identity (dense high-cv products vs light air). The inviscid flux reads local
// gamma = 1 + 1/G (RHS.cpp); here we provide the IC, the contact advection of G,
// and a G-aware pressure diagnostic (the non-oscillatory gate).
struct MultifluidParams {
    bool enabled  = false;
    Real gamma_air = 1.4, gamma_p = 1.25;     // air vs products (larger cv -> lower gamma)
    Real R = 1.0;
    Real rho_p = 10.0, T_p = 100.0;           // dense hot products (used if q<=0)
    Real rho_a = 1.0,  T_a = 1.0;             // light cold air
    Real r0 = 0.1, tanh_thickness = 0.012;    // interface radius + width
    Real Y42_amp = 0.2;                       // (4,2) angular perturbation of r0
    // Chapman-Jouguet IC: if q > 0, the products region is set to the CJ state
    // (rho_CJ, p_CJ, outward u_CJ) of a detonation in the unreacted explosive
    // (density rho_e, temperature T_e) with specific heat release q and the
    // products gamma_p, instead of the arbitrary (rho_p, T_p) blob.
    Real q = 0.0, rho_e = 0.0, T_e = 1.0;
    // Fraction of the CJ particle velocity u_CJ imposed on the products blob at
    // t=0. u_CJ is the post-shock speed behind a *propagating* detonation front,
    // not the uniform velocity of a products bubble at rest. For a confined
    // products-bubble IC, 0 (start at rest, let p drive expansion) is the
    // standard, far less stiff choice; the CJ thermodynamic state (rho_CJ,
    // p_CJ) is still imposed. 1.0 reproduces the full CJ particle velocity.
    Real cj_u_frac = 0.0;
};

// Initialize the two-fluid blast: rho, momentum=0, rhoE consistent with the
// LOCAL gamma, and the G field. Interface (rho, p, G) blended with tanh; the
// interface radius is perturbed by a Y_4^2 mode to seed baroclinic instability.
void mf_init_blast(State& U, Field3D& G, const Grid& g, const MultifluidParams& mp);

// Fill the G ghost cells (slip-wall/outflow -> zero-gradient mirror; periodic ->
// wrap) over the full padded field. Call after mf_init_blast so the first RHS
// sees valid local gamma in the ghosts (serial single-domain BC fill).
void mf_fill_G_bcs(Field3D& G, const BCSet& bc);

// Advect G with the resolved velocity (operator-split, upwind). Slip-wall ->
// Neumann (zero-gradient); periodic -> wrap. Double-buffered. The G ghosts are
// refilled both before (for the upwind gradient) and after the update (so they
// are valid for the next RHS evaluation, which reads local gamma from G ghosts).
void mf_advect_G(Field3D& G, const State& U, const Grid& g, const BCSet& bc, Real dt);

#ifdef BLAST_MPI
// MPI variant: halo-exchange + physical-face BCs for G (interior neighbours via
// Halo, walls via apply_bcs) instead of the serial full-field fill, before and
// after the upwind update. Reads U only at the cell centre, so U ghosts are not
// needed here.
void mf_advect_G(Field3D& G, const State& U, const Grid& g, const BCSet& bc,
                 Real dt, const Domain& d, Halo& halo);
#endif

// G-aware pressure extremes p=(rhoE-ke)/G over the interior, for the
// non-oscillatory gate (uniform-p contact must stay uniform).
void mf_pressure_minmax(const State& U, const Field3D& G, Real& pmin, Real& pmax);

}  // namespace blast
