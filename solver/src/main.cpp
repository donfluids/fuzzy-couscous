#include "bc/BC.hpp"
#include "core/Config.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "diagnostics/FFT.hpp"
#include "diagnostics/Spectra.hpp"
#include "diagnostics/Statistics.hpp"
#include "ic/Canonical.hpp"
#include "io/HDF5Writer.hpp"
#include "io/Log.hpp"
#include "io/Restart.hpp"
#include "numerics/Filter.hpp"
#include "numerics/RHS.hpp"
#include "numerics/RK3.hpp"
#include "physics/EOS.hpp"
#include "physics/Forcing.hpp"
#include "physics/Multifluid.hpp"
#include "turbulence/BHR.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

using namespace blast;

namespace {

void apply_ic(State& U, const Grid& g, const IdealGas& eos, const Config& c) {
    switch (c.ic.type) {
        case ICType::SodX:        ic_sod_x(U, g, eos); break;
        case ICType::ShuOsherX:   ic_shu_osher_x(U, g, eos); break;
        case ICType::Sedov:
            ic_sedov_3d(U, g, eos, /*E*/ 1.0, /*rho*/ 1.0, /*p_amb*/ 1e-5,
                        /*r_blast*/ 4.0 * g.dx());
            break;
        case ICType::TaylorGreen:
            ic_taylor_green_3d(U, g, eos, 1.0, 1.0, 0.1);
            break;
        case ICType::TopHatSphere:
            ic_sphere_blast_3d(U, g, eos, c.ic.rho_B, c.ic.T_B,
                               c.ic.rho_0, c.ic.T_0, c.ic.r0, 0.0,
                               c.ic.Y42_amp, c.ic.ensemble_amp,
                               c.ic.ensemble_seed);
            break;
        case ICType::SmoothSphere:
            ic_sphere_blast_3d(U, g, eos, c.ic.rho_B, c.ic.T_B,
                               c.ic.rho_0, c.ic.T_0, c.ic.r0,
                               (c.ic.tanh_thickness > 0 ? c.ic.tanh_thickness
                                                        : 1.5 * g.dx()),
                               c.ic.Y42_amp, c.ic.ensemble_amp,
                               c.ic.ensemble_seed);
            break;
        case ICType::CJDetonation:
            ic_cj_detonation_3d(U, g, eos, c.ic.rho_0, c.ic.T_0,
                                /*q*/ c.ic.cj_velocity,
                                c.ic.r0,
                                (c.ic.tanh_thickness > 0 ? c.ic.tanh_thickness
                                                          : 1.5 * g.dx()),
                                c.ic.Y42_amp);
            break;
        case ICType::CBC:
            BLAST_ERROR("CBC IC not yet implemented in main driver");
            std::exit(1);
    }
}

ViscousParams to_viscous(const PhysicsConfig& p) {
    ViscousParams vp;
    vp.mu = p.mu;
    vp.prandtl = p.prandtl;
    vp.bulk_visc = p.bulk_visc;
    vp.hyper_coeff = p.hyper_coeff;
    vp.hyper6_coeff = p.hyper6_coeff;
    vp.hyper_method = p.hyper_method;
    return vp;
}

ViscousParams to_viscous(const PhysicsConfig& p, const BCSet& bc) {
    ViscousParams vp = to_viscous(p);
    vp.spectral_bc_mode = bc.all_slip_wall() ? SpectralBCMode::SlipWall
                                             : SpectralBCMode::Periodic;
    return vp;
}

void log_header(std::ostream& os) {
    os << "step,time,dt,KE,tke,u_rms,M_t,c_mean,rho_mean,p_mean,T_mean,"
          "omega2,div2,eps_total,eps_sol,eps_dil,K_sol,K_dil\n";
}

void log_row(std::ostream& os, int step, Real t, Real dt,
             const VelocityStats& s, const DissipationBudget& b,
             const HelmholtzResult& h) {
    os << step << ',' << t << ',' << dt << ','
       << s.ke_total << ',' << s.tke << ',' << s.u_rms << ',' << s.M_t << ','
       << s.c_mean << ',' << s.rho_mean << ',' << s.p_mean << ',' << s.T_mean << ','
       << b.omega2_mean << ',' << b.div2_mean << ','
       << b.eps_total << ',' << b.eps_sol << ',' << b.eps_dil << ','
       << h.K_sol << ',' << h.K_dil << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: blast_les <config.toml>\n";
        return 1;
    }
    const std::string cfg_path = argv[1];

    init_log("blast", "");
    BLAST_INFO("loading config {}", cfg_path);
    Config c;
    try {
        c = load_config(cfg_path);
    } catch (const std::exception& e) {
        BLAST_ERROR("config load failed: {}", e.what());
        return 1;
    }

    BLAST_INFO("grid {}x{}x{} L={}", c.grid.nx, c.grid.ny, c.grid.nz, c.grid.lx);
    BLAST_INFO("t_end={} cfl_hyp={} cfl_vis={}",
               c.time.t_end, c.time.cfl_hyperbolic, c.time.cfl_viscous);

    IdealGas eos{c.physics.eos};
    ViscousParams vp = to_viscous(c.physics, c.bc);
    // Localized artificial diffusivity (LAD), configured via the [afp] block.
    vp.abv_enabled      = c.afp.enabled;
    vp.abv_r            = c.afp.r_order;
    vp.abv_cbeta        = c.afp.C_beta;
    vp.abv_cmu          = c.afp.C_mu;
    vp.abv_ckappa       = c.afp.C_kappa;
    vp.abv_cD           = c.afp.C_D;
    vp.abv_disable_weno = c.afp.disable_weno;
    if (vp.abv_enabled)
        BLAST_INFO("LAD on: C_beta={} C_mu={} C_kappa={} disable_weno={}",
                   vp.abv_cbeta, vp.abv_cmu, vp.abv_ckappa, vp.abv_disable_weno);

    State U(c.grid.nx, c.grid.ny, c.grid.nz);
    Real start_time = 0.0;
    int  start_step = 0;
    if (!c.output.restart_path.empty()) {
        try {
            auto h = read_checkpoint(c.output.restart_path, U, c.grid);
            start_time = h.time;
            start_step = h.step;
            BLAST_INFO("restart: loaded {} at step={} t={}",
                       c.output.restart_path, start_step, start_time);
        } catch (const std::exception& e) {
            BLAST_ERROR("restart failed: {}", e.what());
            return 1;
        }
    } else {
        apply_ic(U, c.grid, eos, c);
    }

    BCSet bc = c.bc;
    apply_bcs(U, bc);

    // Two-gamma multifluid: G = 1/(gamma-1) field carries dense-products vs air.
    Field3D Gfield(c.grid.nx, c.grid.ny, c.grid.nz);
    Gfield.fill(1.0 / (c.physics.eos.gamma - 1.0));
    const Field3D* gptr = nullptr;
    if (c.multifluid.enabled) {
        MultifluidParams mp;
        mp.enabled = true;
        mp.gamma_air = c.physics.eos.gamma;
        mp.gamma_p   = c.multifluid.gamma_p;
        mp.R         = c.physics.eos.R;
        mp.rho_p = c.multifluid.rho_p; mp.T_p = c.multifluid.T_p;
        mp.rho_a = c.multifluid.rho_a; mp.T_a = c.multifluid.T_a;
        mp.q = c.multifluid.q; mp.rho_e = c.multifluid.rho_e; mp.T_e = c.multifluid.T_e;
        mp.cj_u_frac = c.multifluid.cj_u_frac;
        mp.r0 = c.ic.r0; mp.tanh_thickness = c.ic.tanh_thickness; mp.Y42_amp = c.ic.Y42_amp;
        mf_init_blast(U, Gfield, c.grid, mp);
        apply_bcs(U, bc);
        gptr = &Gfield;
        BLAST_INFO("multifluid ON: products gamma={} rho={} T={} vs air gamma={} rho={}",
                   mp.gamma_p, mp.rho_p, mp.T_p, mp.gamma_air, mp.rho_a);
    }

    // BHR variable-density turbulence model (operator-split sub-step).
    BHRParams bp;
    bp.enabled    = c.turbulence.enabled;
    bp.feedback   = c.turbulence.feedback;
    bp.mu_phys    = c.physics.mu;
    bp.C_a        = c.turbulence.C_a;
    bp.C_b        = c.turbulence.C_b;
    bp.L_max      = c.turbulence.L_max;
    bp.prod_limit = c.turbulence.prod_limit;
    bp.seed_scale = c.turbulence.seed_scale;
    bp.b_seed     = c.turbulence.b_seed;
    TurbState turb(c.grid.nx, c.grid.ny, c.grid.nz);
    if (bp.enabled) {
        bhr_init(turb, U, c.grid, eos, bp);
        BLAST_INFO("BHR turbulence model ON (feedback={}, b_seed={})",
                   bp.feedback, bp.b_seed);
    }

    RK3 driver(c.grid.nx, c.grid.ny, c.grid.nz, U.ng());

    std::unique_ptr<SpectralForcing> forcing;
    if (c.forcing.enabled) {
        SpectralForcing::Params fp;
        fp.k_lo = c.forcing.k_lo;
        fp.k_hi = c.forcing.k_hi;
        fp.eps_target = c.forcing.eps_target;
        fp.T_corr = c.forcing.T_corr;
        fp.seed   = c.forcing.seed;
        forcing = std::make_unique<SpectralForcing>(c.grid, fp);
        BLAST_INFO("spectral forcing: {} modes in k in [{}, {}], "
                   "eps_target={:.3e}, T_corr={:.3e}",
                   forcing->num_modes(), fp.k_lo, fp.k_hi,
                   fp.eps_target, fp.T_corr);
    }

    std::filesystem::create_directories(c.output.out_dir);
    HDF5Writer writer(c.output.out_dir, c.run_name);
    std::ofstream stats_file(c.output.out_dir + "/" + c.run_name + "_stats.csv");
    log_header(stats_file);

    FFT3DPlan fft(c.grid.nx, c.grid.ny, c.grid.nz);

    auto write_diagnostics = [&](int step, Real t, Real dt) {
        auto s = velocity_stats(U, eos);
        auto b = dissipation_budget(U, c.grid, eos, vp);
        HelmholtzResult h{};
        ShellSpectrum sp{};
        if (c.output.write_helmholtz || c.output.write_spectra)
            h = helmholtz_decompose(U, c.grid, fft);
        if (c.output.write_spectra) {
            sp = velocity_spectrum(U, c.grid, fft);
            writer.append_spectra(h, sp, t, step);
        }
        log_row(stats_file, step, t, dt, s, b, h);
        stats_file.flush();
        BLAST_INFO("step {:6d} t={:.6e} dt={:.3e} KE={:.4e} tke={:.4e} M_t={:.4f} "
                   "eps_sol={:.3e} eps_dil={:.3e} K_dil/K_sol={:.3e}",
                   step, t, dt, s.ke_total, s.tke, s.M_t,
                   b.eps_sol, b.eps_dil,
                   (h.K_sol > 0 ? h.K_dil / h.K_sol : 0.0));
    };

    Real t = start_time;
    int step = start_step;
    write_diagnostics(step, t, 0.0);
    writer.write_snapshot(U, c.grid, eos, t, step);

    const std::string ckpt_path =
        c.output.out_dir + "/" + c.run_name + ".ckpt.h5";

    while (t < c.time.t_end && step < c.time.max_steps) {
        Real dt_hyp = max_dt_hyperbolic(U, c.grid, eos, c.time.cfl_hyperbolic, gptr);
        Real dt_vis = (vp.mu > 0.0)
                    ? max_dt_viscous(U, c.grid, vp, c.time.cfl_viscous)
                    : 1e30;
        Real dt_turb = 1e30;
        if (bp.enabled && bp.feedback) {
            const Real nut = bhr_max_nu_t(U, turb, bp);
            const Real dxm = std::min({c.grid.dx(), c.grid.dy(), c.grid.dz()});
            if (nut > 0.0) dt_turb = c.time.cfl_viscous * dxm * dxm / nut;
        }
        // LAD viscous limit (one-step lag: uses the max artificial diffusivity
        // from the previous step; 1e30 on the first step).
        Real dt_abv = 1e30;
        if (vp.abv_enabled) {
            const Real num = driver.last_abv_nu_max();
            if (num > 0.0) {
                const Real dxm = std::min({c.grid.dx(), c.grid.dy(), c.grid.dz()});
                dt_abv = c.time.cfl_viscous * dxm * dxm / num;
            }
        }
        Real dt = std::min({dt_hyp, dt_vis, dt_turb, dt_abv, c.time.dt_max});
        if (t + dt > c.time.t_end) dt = c.time.t_end - t;
        if (!std::isfinite(dt) || dt <= 0.0) {
            BLAST_ERROR("non-finite dt at step {}; stopping", step);
            break;
        }

        driver.step(U, c.grid, bc, eos, vp, dt, gptr);
        if (gptr) mf_advect_G(Gfield, U, c.grid, bc, dt);
        if (forcing) {
            forcing->evolve_ou(dt);
            forcing->apply(U, c.grid, dt);
            apply_bcs(U, bc);
        }
        if (bp.enabled) {
            apply_bcs(U, bc);
            bhr_substep(U, turb, c.grid, bc, eos, bp, dt);
        }
        t += dt;
        ++step;

        if (c.filter.enabled && c.filter.every > 0
            && step % c.filter.every == 0) {
            apply_lele_filter(U, bc, c.filter.sigma);
        }

        if (step % c.output.stats_every == 0) {
            write_diagnostics(step, t, dt);
            if (bp.enabled) {
                Real kmx, amx, bmx;
                bhr_peaks(turb, kmx, amx, bmx);
                BLAST_INFO("  BHR <rho k>={:.3e} k_pk={:.3e} |a|_pk={:.3e} b_pk={:.3e}",
                           bhr_tke_integral(U, turb, c.grid), kmx, amx, bmx);
            }
            if (gptr) {
                Real pmn, pmx;
                mf_pressure_minmax(U, Gfield, pmn, pmx);
                BLAST_INFO("  multifluid p in [{:.3e}, {:.3e}]", pmn, pmx);
            }
        }
        if (step % c.output.snapshot_every == 0) {
            writer.write_snapshot(U, c.grid, eos, t, step);
            if (bp.enabled) {
                bhr_write_radial_profiles(U, turb, c.grid,
                    c.output.out_dir + "/" + c.run_name + "_bhrprof_"
                    + std::to_string(step) + ".csv", t);
                Real fk_mean = bhr_write_sensor_profiles(U, turb, c.grid,
                    c.output.out_dir + "/" + c.run_name + "_sensor_"
                    + std::to_string(step) + ".csv", t);
                BLAST_INFO("  hybrid sensor <f_k>={:.3f}", fk_mean);
            }
        }
        if (c.output.checkpoint_every > 0
            && step % c.output.checkpoint_every == 0)
            write_checkpoint(ckpt_path, U, c.grid, t, step);
    }

    BLAST_INFO("finished: step={} t={}", step, t);
    return 0;
}
