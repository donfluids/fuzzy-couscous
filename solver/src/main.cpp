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
#include "numerics/RHS.hpp"
#include "numerics/RK3.hpp"
#include "physics/EOS.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
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
    ViscousParams vp = to_viscous(c.physics);

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

    RK3 driver(c.grid.nx, c.grid.ny, c.grid.nz, U.ng());

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
        Real dt_hyp = max_dt_hyperbolic(U, c.grid, eos, c.time.cfl_hyperbolic);
        Real dt_vis = (vp.mu > 0.0)
                    ? max_dt_viscous(U, c.grid, vp, c.time.cfl_viscous)
                    : 1e30;
        Real dt = std::min({dt_hyp, dt_vis, c.time.dt_max});
        if (t + dt > c.time.t_end) dt = c.time.t_end - t;
        if (!std::isfinite(dt) || dt <= 0.0) {
            BLAST_ERROR("non-finite dt at step {}; stopping", step);
            break;
        }

        driver.step(U, c.grid, bc, eos, vp, dt);
        t += dt;
        ++step;

        if (step % c.output.stats_every == 0)
            write_diagnostics(step, t, dt);
        if (step % c.output.snapshot_every == 0)
            writer.write_snapshot(U, c.grid, eos, t, step);
        if (c.output.checkpoint_every > 0
            && step % c.output.checkpoint_every == 0)
            write_checkpoint(ckpt_path, U, c.grid, t, step);
    }

    BLAST_INFO("finished: step={} t={}", step, t);
    return 0;
}
