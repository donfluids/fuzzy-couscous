#pragma once

#include "core/Config.hpp"   // BCSet
#include "core/Field3D.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "physics/MixtureEOS.hpp"

#ifdef BLAST_MPI
#include "parallel/Domain.hpp"
#include "parallel/Halo.hpp"
#endif

namespace blast {

// Auxiliary state for the true five-equation (Allaire-Kapila-Massoni) model.
// These three fields live ALONGSIDE the NCONS=5 conserved State (which already
// carries the mixture mass rho = Z1+Z2, momentum, and total energy):
//   Z1 = alpha1 * rho1   (partial density, phase 1)  -- conservative
//   Z2 = alpha2 * rho2   (partial density, phase 2)  -- conservative
//   a1 = alpha1          (volume fraction)           -- non-conservative transport
// They are advanced in lockstep with the State inside each RK3 stage (NOT
// operator-split like the two-gamma marker G), because the partial-mass fluxes
// must reuse the same stage velocity / reconstruction as the mixture-mass flux
// so that Z1+Z2 == rho exactly.
struct FiveEqAux {
    Field3D Z1, Z2, a1;
    void allocate(int nx, int ny, int nz, int ng = NGHOST) {
        Z1.resize(nx, ny, nz, ng); Z1.set_name("Z1");
        Z2.resize(nx, ny, nz, ng); Z2.set_name("Z2");
        a1.resize(nx, ny, nz, ng); a1.set_name("alpha1");
    }
};

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

    // ---- JWL products EOS (for real high explosives, e.g. TNT) -------------
    // When jwl_mode, the marker field is a products mass fraction phi in [0,1]
    // (not G), the products region is the tabulated CJ state (rho_cj, p_cj) with
    // the JWL EOS, and ambient air is (rho_a, p_a). All quantities here are
    // already NONDIMENSIONAL (the caller divides by rho_ref/p_ref at load).
    bool      jwl_mode  = false;
    JWLParams jwl{};
    Real      rho_cj = 0.0, p_cj = 0.0, p_a = 0.0;
    Real      phi_switch = 0.5;

    // ---- Five-equation diffuse-interface mode -----------------------------
    // Two phases, each a stiffened gas or a JWL EOS (see PhaseEOS). The IC is a
    // tanh-blended bubble (radius r0, like the two-gamma blast): volume fraction
    // a1 goes from a1_out (ambient) to a1_in (bubble), with phase densities
    // rho1/rho2 and pressures p_out/p_in, optionally translating at (u0,v0,w0).
    bool     five_eq_mode = false;
    PhaseEOS ph[2];
    Real     a1_in = 1.0 - 1e-6, a1_out = 1e-6;
    Real     rho1 = 1.0, rho2 = 1.0;
    Real     p_in = 1.0, p_out = 1.0;
    Real     u0 = 0.0, v0 = 0.0, w0 = 0.0;

    // Build the matching mixture EOS for the flux/CFL loops.
    MixtureEOS mixture() const {
        MixtureEOS m;
        m.gamma_air = gamma_air;
        if (five_eq_mode) {
            m.mode = MixMode::FiveEquation;
            m.phase[0] = ph[0];
            m.phase[1] = ph[1];
        } else if (jwl_mode) {
            m.mode = MixMode::JWL;
            m.phi_switch = phi_switch;
            m.jwl = jwl;
        } else {
            m.mode = MixMode::TwoGamma;
        }
        return m;
    }
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

// Marker-aware pressure extremes over the interior, for the non-oscillatory gate
// (uniform-p contact must stay uniform) and diagnostics. mix==nullptr keeps the
// two-gamma form p=(rhoE-ke)/G; with a JWL mix the products use the JWL EOS.
void mf_pressure_minmax(const State& U, const Field3D& G, Real& pmin, Real& pmax,
                        const MixtureEOS* mix = nullptr);

// G boundedness / mixing diagnostic. G is an advected marker, so it must stay
// within its initial range [G_air, G_products] (discrete maximum principle for
// monotone upwind at CFL<=1); gmin/gmax leaving that range flags instability.
// gvar = <(G-<G>)^2> measures interface mixing (bounded above by the initial
// range, dissipated by the upwind scheme); growth would flag instability.
struct GStats { Real gmin, gmax, gmean, gvar; };
GStats mf_g_stats(const Field3D& G);
#ifdef BLAST_MPI
GStats mf_g_stats(const Field3D& G, long long N_global, MPI_Comm comm);
#endif

// ---- Five-equation helpers ------------------------------------------------

// Initialize the five-equation state: a tanh-blended bubble (volume fraction,
// phase densities, pressure) consistent with the mixture EOS so the flux-loop
// pressure matches the IC exactly. Sets U (rho=Z1+Z2, momentum, rhoE) and aux.
void mf_init_5eq(State& U, FiveEqAux& aux, const Grid& g,
                 const MultifluidParams& mp);

// Fill ghost cells of all three aux fields (zero-gradient at walls/outflow,
// wrap at periodic), reusing the marker BC rule. Call each RK stage so the
// aux flux sees valid ghosts.
void mf_fill_aux_bcs(FiveEqAux& aux, const BCSet& bc);

// SSP-RK3 combiners for the aux bundle, mirroring state_axpby / state_axpbypcz
// (interior + ghosts). out = a*x + b*y (+ c*z).
void aux_axpby(FiveEqAux& out, Real a, const FiveEqAux& x, Real b,
               const FiveEqAux& y);
void aux_axpbypcz(FiveEqAux& out, Real a, const FiveEqAux& x, Real b,
                  const FiveEqAux& y, Real c, const FiveEqAux& z);

// Boundedness for the five-equation state (interior only): clamp 0<=a1<=1 with
// a floor that keeps both phases present, clamp Z1,Z2 positive, and resync
// U[RHO] = Z1+Z2 so the mixture mass stays consistent. Returns cells clamped.
long enforce_5eq_bounds(State& U, FiveEqAux& aux, Real a_floor, Real z_floor);

}  // namespace blast
