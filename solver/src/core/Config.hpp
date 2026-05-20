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

struct AFPConfig {
    bool enabled  = true;
    int  r_order  = 4;
    Real C_mu     = 0.002;
    Real C_beta   = 1.0;
    Real C_kappa  = 0.01;
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
    OutputConfig   output;
    std::string    run_name = "run";
};

Config load_config(const std::string& path);

}  // namespace blast
