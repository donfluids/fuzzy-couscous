#pragma once

#include "core/Grid.hpp"
#include "core/Types.hpp"
#include "physics/ViscousFlux.hpp"

#include <string>

namespace blast {

enum class BCType { Periodic, SlipWall, Outflow };

enum class ICType {
    SodX, ShuOsherX, Sedov, TaylorGreen, CBC,
    TopHatSphere, SmoothSphere, CJDetonation
};

struct PhysicsConfig {
    GammaLaw eos;
    Real     mu = 1.8e-5;
    Real     prandtl = 0.71;
    Real     bulk_visc = 0.0;
    // Hyperdissipation coefficient nu_h for the -(nabla^2)^2 U sink.
    // <=0 disables. See ViscousParams::hyper_coeff for the rationale.
    Real     hyper_coeff = 0.0;
    // Higher-power hyperdissipation coefficient nu_h6 for the
    // +(nabla^2)^3 U sink. <=0 disables. See ViscousParams::hyper6_coeff.
    Real     hyper6_coeff = 0.0;
    // Discretization for both hyperdissipation terms. "fd" (default) uses
    // composed Laplacian stencils; "spectral" uses an exact FFT operator
    // and requires all-periodic BCs (validated at config load) and serial.
    HyperMethod hyper_method = HyperMethod::FiniteDifference;
};

// Localized artificial diffusivity ("artificial fluid properties", Kawai-Lele).
// Opt-in shock/contact dissipation that lets the central scheme run without
// WENO. Off by default so existing runs are unchanged. NOTE: the finite-
// difference path uses a 2nd-derivative sensor (r=2) to fit NGHOST=6; r_order
// is reserved for a future wider-halo 4th-derivative variant.
struct AFPConfig {
    bool enabled      = false;
    int  r_order      = 4;
    Real C_mu         = 0.002;   // artificial shear
    Real C_beta       = 1.0;     // artificial bulk (shocks)
    Real C_kappa      = 0.01;    // artificial thermal conductivity (entropy/contacts)
    Real C_D          = 0.01;    // artificial mass diffusivity (density contacts)
    bool disable_weno = false;   // suppress Ducros/WENO so LAD is the sole shock sink
};

struct TimeConfig {
    Real cfl_hyperbolic = 0.5;
    Real cfl_viscous    = 0.25;
    Real t_end          = 0.0;
    Real dt_max         = 1e30;
    int  max_steps      = 1'000'000;
};

struct FilterConfig {
    // 6th-order Lele explicit low-pass filter applied every `every` steps to
    // each conserved variable (interior + 3 ghost cells per face). Strength
    // sigma in [0, 1]: U_new = (1 - sigma) U + sigma F(U). For LES of
    // isotropic turbulence on under-resolved grids, sigma ~ 0.1-0.25 every
    // 5-10 steps is the standard recipe (Visbal & Gaitonde 2002).
    bool enabled = false;
    int  every   = 0;
    Real sigma   = 0.2;
};

struct BCSet {
    BCType xlo = BCType::Periodic, xhi = BCType::Periodic;
    BCType ylo = BCType::Periodic, yhi = BCType::Periodic;
    BCType zlo = BCType::Periodic, zhi = BCType::Periodic;

    bool all_periodic() const {
        return xlo == BCType::Periodic && xhi == BCType::Periodic
            && ylo == BCType::Periodic && yhi == BCType::Periodic
            && zlo == BCType::Periodic && zhi == BCType::Periodic;
    }
    bool all_slip_wall() const {
        return xlo == BCType::SlipWall && xhi == BCType::SlipWall
            && ylo == BCType::SlipWall && yhi == BCType::SlipWall
            && zlo == BCType::SlipWall && zhi == BCType::SlipWall;
    }
};

struct ICParams {
    ICType type = ICType::SodX;
    // Sphere blast (top-hat / smoothed / Y4,2-perturbed).
    Real r0       = 0.0;
    Real rho_B    = 1.0;
    Real T_B      = 1.0;
    Real rho_0    = 1.0;
    Real T_0      = 1.0;
    Real Y42_amp  = 0.0;
    Real tanh_thickness = 0.0;
    // CJ detonation extras.
    Real cj_velocity = 0.0;
    // CBC / Rogallo random.
    Real cbc_urms    = 1.0;
    Real cbc_k_peak  = 4.0;
    int  cbc_seed    = 12345;
    // Ensemble support: when ensemble_amp > 0, sphere_blast / cj_detonation
    // ICs add a random multi-mode angular perturbation drawn from this seed
    // on top of the Y_{4,2} pattern.  Different seeds -> distinguishable
    // realizations of the same physical setup (reviewer M6).
    Real ensemble_amp  = 0.0;
    int  ensemble_seed = 0;
};

struct ForcingConfig {
    // Eswaran-Pope stochastic spectral OU forcing for statistically
    // stationary HIT. Off by default; switch on with `enabled = true`
    // and set eps_target.
    bool enabled    = false;
    int  k_lo       = 1;
    int  k_hi       = 3;
    Real eps_target = 0.1;
    Real T_corr     = 1.0;
    int  seed       = 12345;
};

struct MultifluidConfig {
    bool enabled   = false;    // two-gamma multifluid (dense products vs air)
    Real gamma_p   = 1.25;     // products gamma (larger cv); air uses physics.eos.gamma
    Real rho_p     = 10.0;
    Real T_p       = 100.0;
    Real rho_a     = 1.0;
    Real T_a       = 1.0;
    // Chapman-Jouguet products IC (q>0 enables it; overrides rho_p/T_p):
    Real q         = 0.0;      // specific heat release of the explosive
    Real rho_e     = 0.0;      // unreacted explosive density (0 -> use rho_p)
    Real T_e       = 1.0;      // unreacted explosive temperature
    Real cj_u_frac = 0.0;      // fraction of CJ particle velocity imposed at t=0
};

struct TurbulenceConfig {
    bool enabled  = false;     // run the BHR sub-step
    bool feedback = false;     // two-way coupling to the mean
    Real C_a = 1.0, C_b = 1.0;
    Real L_max = 0.5, prod_limit = 10.0;
    Real seed_scale = 1e-3, b_seed = 0.0;
};

struct OutputConfig {
    std::string out_dir         = "out";
    int         snapshot_every  = 100;
    int         stats_every     = 10;
    int         checkpoint_every = 0;   // 0 disables
    bool        write_spectra   = true;
    bool        write_helmholtz = true;
    std::string restart_path    = "";   // empty -> cold start
};

struct Config {
    Grid           grid;
    BCSet          bc;
    ICParams       ic;
    PhysicsConfig  physics;
    AFPConfig      afp;
    TimeConfig     time;
    FilterConfig   filter;
    ForcingConfig  forcing;
    TurbulenceConfig turbulence;
    MultifluidConfig multifluid;
    OutputConfig   output;
    std::string    run_name = "run";
};

Config load_config(const std::string& path);

}  // namespace blast
