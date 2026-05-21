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
#include "parallel/Domain.hpp"
#include "parallel/Halo.hpp"
#include "physics/EOS.hpp"
#include "physics/Forcing.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

using namespace blast;

namespace {

void apply_ic_local(State& U, const Grid& local_g, const Grid& global_g,
                    const IdealGas& eos, const Config& c) {
    const Real xc_g = global_g.x0 + 0.5 * global_g.lx;
    const Real yc_g = global_g.y0 + 0.5 * global_g.ly;
    const Real zc_g = global_g.z0 + 0.5 * global_g.lz;
    switch (c.ic.type) {
        case ICType::TaylorGreen:
            ic_taylor_green_3d(U, local_g, eos, 1.0, 1.0, 0.1);
            break;
        case ICType::TopHatSphere:
            ic_sphere_blast_3d(U, local_g, eos, c.ic.rho_B, c.ic.T_B,
                               c.ic.rho_0, c.ic.T_0, c.ic.r0, 0.0,
                               c.ic.Y42_amp, c.ic.ensemble_amp, c.ic.ensemble_seed,
                               xc_g, yc_g, zc_g);
            break;
        case ICType::SmoothSphere:
            ic_sphere_blast_3d(U, local_g, eos, c.ic.rho_B, c.ic.T_B,
                               c.ic.rho_0, c.ic.T_0, c.ic.r0,
                               (c.ic.tanh_thickness > 0 ? c.ic.tanh_thickness
                                                        : 1.5 * local_g.dx()),
                               c.ic.Y42_amp, c.ic.ensemble_amp, c.ic.ensemble_seed,
                               xc_g, yc_g, zc_g);
            break;
        case ICType::CJDetonation:
            ic_cj_detonation_3d(U, local_g, eos, c.ic.rho_0, c.ic.T_0,
                                c.ic.cj_velocity, c.ic.r0,
                                (c.ic.tanh_thickness > 0 ? c.ic.tanh_thickness
                                                          : 1.5 * local_g.dx()),
                                c.ic.Y42_amp, xc_g, yc_g, zc_g);
            break;
        case ICType::GaussianBlast:
            ic_gaussian_blast_3d(U, local_g, eos, c.ic.blast_energy, c.ic.r0,
                                 c.ic.rho_0, c.ic.T_0, c.ic.Y42_amp,
                                 c.ic.ensemble_amp, c.ic.ensemble_seed,
                                 xc_g, yc_g, zc_g);
            break;
        case ICType::SodX:
        case ICType::ShuOsherX:
        case ICType::Sedov:
        case ICType::CBC:
            BLAST_ERROR("IC type not yet supported in MPI driver");
            MPI_Abort(MPI_COMM_WORLD, 1);
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

}  // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "error: MPI implementation does not support FUNNELED\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 2) {
        if (world_rank == 0)
            std::cerr << "usage: blast_les_mpi <config.toml>\n";
        MPI_Finalize();
        return 1;
    }
    const std::string cfg_path = argv[1];

    init_log("blast_mpi", "");
    if (world_rank == 0) BLAST_INFO("loading config {}", cfg_path);

    Config c;
    try { c = load_config(cfg_path); }
    catch (const std::exception& e) {
        if (world_rank == 0) BLAST_ERROR("config load failed: {}", e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }


    // Nested scope: all MPI-owning objects must destruct BEFORE MPI_Finalize.
    {
    Grid global_g = c.grid;
    Domain domain(MPI_COMM_WORLD, global_g, c.bc);
    Grid local_g = domain.local_grid(global_g);

    if (world_rank == 0) {
        BLAST_INFO("MPI: world_size={} dims={}x{}x{} global={}x{}x{}",
                   world_size,
                   domain.dims()[0], domain.dims()[1], domain.dims()[2],
                   global_g.nx, global_g.ny, global_g.nz);
        BLAST_INFO("local rank-0 grid: {}x{}x{} starting at ({}, {}, {})",
                   local_g.nx, local_g.ny, local_g.nz,
                   local_g.x0, local_g.y0, local_g.z0);
    }

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

    State U(local_g.nx, local_g.ny, local_g.nz);
    Real start_time = 0.0;
    int  start_step = 0;
    if (!c.output.restart_path.empty()) {
        try {
            auto h = read_checkpoint(c.output.restart_path, U, global_g, domain);
            start_time = h.time;
            start_step = h.step;
            if (world_rank == 0)
                BLAST_INFO("restart(MPI): loaded {} at step={} t={}",
                           c.output.restart_path, start_step, start_time);
        } catch (const std::exception& e) {
            if (world_rank == 0) BLAST_ERROR("restart failed: {}", e.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else {
        apply_ic_local(U, local_g, global_g, eos, c);
    }

    BCSet bc = c.bc;
    Halo halo(U, domain);
    halo.exchange(U);
    apply_bcs(U, bc, domain);

    RK3 driver(local_g.nx, local_g.ny, local_g.nz, U.ng());
    if (vp.hyper_method == HyperMethod::Pseudospectral) {
        driver.init_spectral_hyper_mpi(global_g, domain, vp.spectral_bc_mode);
    }

    // Spectral OU forcing (shared across ranks, same seed -> bit-exact MPI).
    // Constructed against the GLOBAL grid so all ranks share the same modes.
    std::unique_ptr<SpectralForcing> forcing;
    if (c.forcing.enabled) {
        SpectralForcing::Params fp;
        fp.k_lo = c.forcing.k_lo;
        fp.k_hi = c.forcing.k_hi;
        fp.eps_target = c.forcing.eps_target;
        fp.T_corr = c.forcing.T_corr;
        fp.seed   = c.forcing.seed;
        forcing = std::make_unique<SpectralForcing>(global_g, fp);
        if (world_rank == 0)
            BLAST_INFO("spectral forcing: {} modes in k in [{}, {}], "
                       "eps_target={:.3e}, T_corr={:.3e}",
                       forcing->num_modes(), fp.k_lo, fp.k_hi,
                       fp.eps_target, fp.T_corr);
    }

    if (world_rank == 0)
        std::filesystem::create_directories(c.output.out_dir);
    MPI_Barrier(MPI_COMM_WORLD);

    HDF5Writer writer(c.output.out_dir, c.run_name);
    writer.set_domain(&domain);

    std::ofstream stats_file;
    if (world_rank == 0) {
        stats_file.open(c.output.out_dir + "/" + c.run_name + "_stats.csv");
        stats_file << "step,time,dt,KE,tke,u_rms,M_t,c_mean,rho_mean,p_mean,"
                     "T_mean,omega2,div2,eps_total,eps_sol,eps_dil,e_total,e_int\n";
    }

    const long long N_global = static_cast<long long>(global_g.nx) * global_g.ny * global_g.nz;

    // Spectra plans. Pick basis from BC configuration:
    //   - all periodic  -> Fourier r2c (distributed FFTW3-MPI)
    //   - all slip-wall -> 3D real-to-real (DCT/DST mix per velocity component)
    // Mixed BCs are currently not supported for spectra (the basis would have
    // to vary per axis); spectra are skipped with a warning in that case.
    const bool periodic_spec = c.bc.all_periodic();
    const bool slip_spec     = c.bc.all_slip_wall();
    const bool any_spec_out  = c.output.write_spectra || c.output.write_helmholtz;
    if (any_spec_out && !periodic_spec && !slip_spec && world_rank == 0) {
        BLAST_WARN("write_spectra/write_helmholtz set but BCs are neither "
                   "all-periodic nor all-slip-wall; spectra disabled");
    }
    std::unique_ptr<FFT3DPlanMPI> fft_plan;
    if (any_spec_out && periodic_spec)
        fft_plan = std::make_unique<FFT3DPlanMPI>(
            global_g.nx, global_g.ny, global_g.nz, domain.comm());

    std::unique_ptr<R2R3DPlanMPI> dct_plan_u, dct_plan_v, dct_plan_w;
    if (any_spec_out && slip_spec) {
        // R2R3DPlanMPI takes kinds outer-to-inner = (z, y, x).
        // u is DST on x (its normal), DCT on y, z;  v is DST on y; w is DST on z.
        dct_plan_u = std::make_unique<R2R3DPlanMPI>(
            global_g.nx, global_g.ny, global_g.nz, domain.comm(),
            r2r::DCT_II, r2r::DCT_II, r2r::DST_II);
        dct_plan_v = std::make_unique<R2R3DPlanMPI>(
            global_g.nx, global_g.ny, global_g.nz, domain.comm(),
            r2r::DCT_II, r2r::DST_II, r2r::DCT_II);
        dct_plan_w = std::make_unique<R2R3DPlanMPI>(
            global_g.nx, global_g.ny, global_g.nz, domain.comm(),
            r2r::DST_II, r2r::DCT_II, r2r::DCT_II);
    }

    auto log_diagnostics = [&](int step, Real t, Real dt) {
        auto s = velocity_stats(U, eos, N_global, domain.comm());
        auto b = dissipation_budget(U, local_g, eos, vp, N_global, domain.comm());
        HelmholtzResult h{};
        ShellSpectrum sp{};
        if (c.output.write_helmholtz || c.output.write_spectra) {
            if (periodic_spec)
                h = helmholtz_decompose_mpi_dist(U, global_g, *fft_plan, domain);
            else if (slip_spec)
                h = helmholtz_decompose_dct_mpi(U, global_g,
                                                *dct_plan_u, *dct_plan_v, *dct_plan_w,
                                                domain);
        }
        if (c.output.write_spectra) {
            if (periodic_spec) {
                sp = velocity_spectrum_mpi_dist(U, global_g, *fft_plan, domain);
            } else if (slip_spec) {
                // helmholtz_decompose_dct_mpi already produced E_sol+E_dil
                // bins of total kinetic energy; reuse for the total spectrum
                // to avoid re-running the 3 forward transforms.
                sp.k = h.E_sol.k;
                sp.E.assign(h.E_sol.E.size(), 0.0);
                for (std::size_t b = 0; b < sp.E.size(); ++b)
                    sp.E[b] = h.E_sol.E[b] + h.E_dil.E[b];
            }
            if (world_rank == 0 && (periodic_spec || slip_spec))
                writer.append_spectra(h, sp, t, step);
        }
        if (world_rank == 0) {
            stats_file << step << ',' << t << ',' << dt << ','
                       << s.ke_total << ',' << s.tke << ',' << s.u_rms << ',' << s.M_t << ','
                       << s.c_mean << ',' << s.rho_mean << ',' << s.p_mean << ','
                       << s.T_mean << ',' << b.omega2_mean << ',' << b.div2_mean << ','
                       << b.eps_total << ',' << b.eps_sol << ',' << b.eps_dil << ','
                       << s.e_total << ',' << s.e_int << '\n';
            stats_file.flush();
            BLAST_INFO("step {:6d} t={:.6e} dt={:.3e} KE={:.4e} tke={:.4e} M_t={:.4f} "
                       "eps_sol={:.3e} eps_dil={:.3e} K_dil/K_sol={:.3e}",
                       step, t, dt, s.ke_total, s.tke, s.M_t,
                       b.eps_sol, b.eps_dil,
                       (h.K_sol > 0 ? h.K_dil / h.K_sol : 0.0));
        }
    };

    Real t = start_time;
    int step = start_step;
    log_diagnostics(step, t, 0.0);
    writer.write_snapshot(U, global_g, eos, t, step);

    const std::string ckpt_path =
        c.output.out_dir + "/" + c.run_name + ".ckpt.h5";

    while (t < c.time.t_end && step < c.time.max_steps) {
        Real dt_hyp = max_dt_hyperbolic(U, local_g, eos, c.time.cfl_hyperbolic,
                                         domain.comm());
        Real dt_vis = (vp.mu > 0.0)
                    ? max_dt_viscous(U, local_g, vp, c.time.cfl_viscous, domain.comm())
                    : 1e30;
        // LAD viscous limit (one-step lag), reduced across ranks.
        Real dt_abv = 1e30;
        if (vp.abv_enabled) {
            Real num_local = driver.last_abv_nu_max();
            Real num = num_local;
            MPI_Allreduce(&num_local, &num, 1, MPI_DOUBLE, MPI_MAX, domain.comm());
            if (num > 0.0) {
                const Real dxm = std::min({local_g.dx(), local_g.dy(), local_g.dz()});
                dt_abv = c.time.cfl_viscous * dxm * dxm / num;
            }
        }
        Real dt = std::min({dt_hyp, dt_vis, dt_abv, c.time.dt_max});
        if (t + dt > c.time.t_end) dt = c.time.t_end - t;
        if (!std::isfinite(dt) || dt <= 0.0) {
            if (world_rank == 0) BLAST_ERROR("non-finite dt at step {}", step);
            break;
        }

        driver.step_mpi(U, local_g, bc, eos, vp, dt, domain, halo);
        if (forcing) {
            forcing->evolve_ou(dt);
            forcing->apply(U, local_g, dt, domain.comm());
            halo.exchange(U);
            apply_bcs(U, bc, domain);
        }
        t += dt;
        ++step;

        if (step % c.output.stats_every == 0)   log_diagnostics(step, t, dt);
        if (step % c.output.snapshot_every == 0)
            writer.write_snapshot(U, global_g, eos, t, step);
        if (c.output.checkpoint_every > 0
            && step % c.output.checkpoint_every == 0)
            write_checkpoint(ckpt_path, U, global_g, t, step, domain);
    }

    if (world_rank == 0)
        BLAST_INFO("finished: step={} t={}", step, t);
    }   // close nested scope -- Halo / Domain destructors fire here
    MPI_Finalize();
    return 0;
}
