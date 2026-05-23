#include "core/Config.hpp"

#include <toml++/toml.hpp>

#include <stdexcept>
#include <string>
#include <string_view>

namespace blast {

namespace {

BCType parse_bc(std::string_view s) {
    if (s == "periodic") return BCType::Periodic;
    if (s == "slip_wall" || s == "slip") return BCType::SlipWall;
    if (s == "outflow")  return BCType::Outflow;
    throw std::invalid_argument("Unknown BC: " + std::string(s));
}

HyperMethod parse_hyper_method(std::string_view s) {
    if (s == "fd" || s == "finite_difference") return HyperMethod::FiniteDifference;
    if (s == "spectral" || s == "pseudospectral") return HyperMethod::Pseudospectral;
    throw std::invalid_argument("Unknown hyper_method: " + std::string(s));
}

ICType parse_ic(std::string_view s) {
    if (s == "sod_x")          return ICType::SodX;
    if (s == "shu_osher_x")    return ICType::ShuOsherX;
    if (s == "sedov")          return ICType::Sedov;
    if (s == "taylor_green")   return ICType::TaylorGreen;
    if (s == "cbc")            return ICType::CBC;
    if (s == "tophat_sphere")  return ICType::TopHatSphere;
    if (s == "smooth_sphere")  return ICType::SmoothSphere;
    if (s == "cj_detonation")  return ICType::CJDetonation;
    if (s == "gaussian_blast") return ICType::GaussianBlast;
    throw std::invalid_argument("Unknown IC: " + std::string(s));
}

template <typename T, typename V>
T pick(const V& v, T fallback) {
    if (auto x = v.template value<T>()) return *x;
    return fallback;
}

}  // namespace

Config load_config(const std::string& path) {
    toml::table tbl;
    try {
        tbl = toml::parse_file(path);
    } catch (const toml::parse_error& e) {
        throw std::runtime_error(std::string("Failed to parse ") + path + ": " + e.what());
    }

    Config c;
    c.run_name = pick<std::string>(tbl["run_name"], c.run_name);

    auto g = tbl["grid"];
    c.grid.nx = pick<int64_t>(g["nx"], c.grid.nx);
    c.grid.ny = pick<int64_t>(g["ny"], c.grid.ny);
    c.grid.nz = pick<int64_t>(g["nz"], c.grid.nz);
    c.grid.lx = pick<double>(g["lx"], c.grid.lx);
    c.grid.ly = pick<double>(g["ly"], c.grid.ly);
    c.grid.lz = pick<double>(g["lz"], c.grid.lz);
    c.grid.x0 = pick<double>(g["x0"], c.grid.x0);
    c.grid.y0 = pick<double>(g["y0"], c.grid.y0);
    c.grid.z0 = pick<double>(g["z0"], c.grid.z0);

    auto bc = tbl["bc"];
    if (auto s = bc["xlo"].value<std::string>()) c.bc.xlo = parse_bc(*s);
    if (auto s = bc["xhi"].value<std::string>()) c.bc.xhi = parse_bc(*s);
    if (auto s = bc["ylo"].value<std::string>()) c.bc.ylo = parse_bc(*s);
    if (auto s = bc["yhi"].value<std::string>()) c.bc.yhi = parse_bc(*s);
    if (auto s = bc["zlo"].value<std::string>()) c.bc.zlo = parse_bc(*s);
    if (auto s = bc["zhi"].value<std::string>()) c.bc.zhi = parse_bc(*s);

    auto ic = tbl["ic"];
    if (auto s = ic["type"].value<std::string>()) c.ic.type = parse_ic(*s);
    c.ic.r0      = pick<double>(ic["r0"],      c.ic.r0);
    c.ic.rho_B   = pick<double>(ic["rho_B"],   c.ic.rho_B);
    c.ic.blast_energy = pick<double>(ic["blast_energy"], c.ic.blast_energy);
    c.ic.T_B     = pick<double>(ic["T_B"],     c.ic.T_B);
    c.ic.rho_0   = pick<double>(ic["rho_0"],   c.ic.rho_0);
    c.ic.T_0     = pick<double>(ic["T_0"],     c.ic.T_0);
    c.ic.Y42_amp = pick<double>(ic["Y42_amp"], c.ic.Y42_amp);
    c.ic.tanh_thickness = pick<double>(ic["tanh_thickness"], c.ic.tanh_thickness);
    c.ic.cj_velocity    = pick<double>(ic["cj_velocity"], c.ic.cj_velocity);
    c.ic.cbc_urms       = pick<double>(ic["cbc_urms"], c.ic.cbc_urms);
    c.ic.cbc_k_peak     = pick<double>(ic["cbc_k_peak"], c.ic.cbc_k_peak);
    c.ic.cbc_seed       = pick<int64_t>(ic["cbc_seed"], c.ic.cbc_seed);
    c.ic.ensemble_amp   = pick<double>(ic["ensemble_amp"], c.ic.ensemble_amp);
    c.ic.ensemble_seed  = pick<int64_t>(ic["ensemble_seed"], c.ic.ensemble_seed);

    auto ph = tbl["physics"];
    c.physics.eos.gamma  = pick<double>(ph["gamma"],  c.physics.eos.gamma);
    c.physics.eos.R      = pick<double>(ph["R"],      c.physics.eos.R);
    c.physics.mu         = pick<double>(ph["mu"],     c.physics.mu);
    c.physics.prandtl    = pick<double>(ph["prandtl"], c.physics.prandtl);
    c.physics.bulk_visc  = pick<double>(ph["bulk_visc"], c.physics.bulk_visc);
    c.physics.hyper_coeff  = pick<double>(ph["hyper_coeff"],  c.physics.hyper_coeff);
    c.physics.hyper6_coeff = pick<double>(ph["hyper6_coeff"], c.physics.hyper6_coeff);
    if (auto s = ph["hyper_method"].value<std::string>())
        c.physics.hyper_method = parse_hyper_method(*s);
    if (auto s = ph["flux_scheme"].value<std::string>()) {
        if (*s == "compact10")     c.physics.flux_compact10 = true;
        else if (*s == "central6") c.physics.flux_compact10 = false;
        else throw std::runtime_error(
            "flux_scheme must be \"central6\" or \"compact10\"");
    }
    if (c.physics.flux_compact10 && c.bc.all_periodic()) {
        throw std::runtime_error(
            "flux_scheme = \"compact10\" uses a slip-wall boundary closure and "
            "is not supported for all-periodic BCs yet");
    }

    if (c.physics.hyper_method == HyperMethod::Pseudospectral
        && !c.bc.all_periodic() && !c.bc.all_slip_wall()) {
        throw std::runtime_error(
            "hyper_method = \"spectral\" requires every axis to be "
            "uniformly \"periodic\" or uniformly \"slip_wall\" "
            "(mixed BCs are not supported)");
    }

    auto a = tbl["afp"];
    c.afp.enabled = pick<bool>(a["enabled"], c.afp.enabled);
    c.afp.r_order = pick<int64_t>(a["r_order"], c.afp.r_order);
    c.afp.C_mu    = pick<double>(a["C_mu"], c.afp.C_mu);
    c.afp.C_beta  = pick<double>(a["C_beta"], c.afp.C_beta);
    c.afp.C_kappa = pick<double>(a["C_kappa"], c.afp.C_kappa);
    c.afp.C_D     = pick<double>(a["C_D"], c.afp.C_D);
    c.afp.disable_weno = pick<bool>(a["disable_weno"], c.afp.disable_weno);

    auto t = tbl["time"];
    c.time.cfl_hyperbolic = pick<double>(t["cfl_hyperbolic"], c.time.cfl_hyperbolic);
    c.time.cfl_viscous    = pick<double>(t["cfl_viscous"], c.time.cfl_viscous);
    c.time.t_end          = pick<double>(t["t_end"],   c.time.t_end);
    c.time.dt_max         = pick<double>(t["dt_max"],  c.time.dt_max);
    c.time.max_steps      = pick<int64_t>(t["max_steps"], c.time.max_steps);

    auto fl = tbl["filter"];
    c.filter.enabled = pick<bool>(fl["enabled"], c.filter.enabled);
    c.filter.every   = pick<int64_t>(fl["every"], c.filter.every);
    c.filter.sigma   = pick<double>(fl["sigma"], c.filter.sigma);

    auto fc = tbl["forcing"];
    c.forcing.enabled    = pick<bool>(fc["enabled"],       c.forcing.enabled);
    c.forcing.k_lo       = pick<int64_t>(fc["k_lo"],       c.forcing.k_lo);
    c.forcing.k_hi       = pick<int64_t>(fc["k_hi"],       c.forcing.k_hi);
    c.forcing.eps_target = pick<double>(fc["eps_target"],  c.forcing.eps_target);
    c.forcing.T_corr     = pick<double>(fc["T_corr"],      c.forcing.T_corr);
    c.forcing.seed       = pick<int64_t>(fc["seed"],       c.forcing.seed);

    auto tu = tbl["turbulence"];
    c.turbulence.enabled    = pick<bool>(tu["enabled"],    c.turbulence.enabled);
    c.turbulence.feedback   = pick<bool>(tu["feedback"],   c.turbulence.feedback);
    c.turbulence.C_a        = pick<double>(tu["C_a"],      c.turbulence.C_a);
    c.turbulence.C_b        = pick<double>(tu["C_b"],      c.turbulence.C_b);
    c.turbulence.L_max      = pick<double>(tu["L_max"],    c.turbulence.L_max);
    c.turbulence.prod_limit = pick<double>(tu["prod_limit"], c.turbulence.prod_limit);
    c.turbulence.seed_scale = pick<double>(tu["seed_scale"], c.turbulence.seed_scale);
    c.turbulence.b_seed     = pick<double>(tu["b_seed"],   c.turbulence.b_seed);

    auto mf = tbl["multifluid"];
    c.multifluid.enabled = pick<bool>(mf["enabled"],   c.multifluid.enabled);
    c.multifluid.conservative = pick<bool>(mf["conservative"], c.multifluid.conservative);
    c.multifluid.eos     = pick<std::string>(mf["eos"], c.multifluid.eos);
    c.multifluid.gamma_p = pick<double>(mf["gamma_p"], c.multifluid.gamma_p);
    c.multifluid.rho_p   = pick<double>(mf["rho_p"],   c.multifluid.rho_p);
    c.multifluid.T_p     = pick<double>(mf["T_p"],     c.multifluid.T_p);
    c.multifluid.rho_a   = pick<double>(mf["rho_a"],   c.multifluid.rho_a);
    c.multifluid.T_a     = pick<double>(mf["T_a"],     c.multifluid.T_a);
    c.multifluid.q       = pick<double>(mf["q"],       c.multifluid.q);
    c.multifluid.rho_e   = pick<double>(mf["rho_e"],   c.multifluid.rho_e);
    c.multifluid.T_e     = pick<double>(mf["T_e"],     c.multifluid.T_e);
    c.multifluid.cj_u_frac = pick<double>(mf["cj_u_frac"], c.multifluid.cj_u_frac);
    // JWL products EOS (eos = "jwl").
    c.multifluid.jwl_A     = pick<double>(mf["jwl_A"],     c.multifluid.jwl_A);
    c.multifluid.jwl_B     = pick<double>(mf["jwl_B"],     c.multifluid.jwl_B);
    c.multifluid.jwl_R1    = pick<double>(mf["jwl_R1"],    c.multifluid.jwl_R1);
    c.multifluid.jwl_R2    = pick<double>(mf["jwl_R2"],    c.multifluid.jwl_R2);
    c.multifluid.jwl_omega = pick<double>(mf["jwl_omega"], c.multifluid.jwl_omega);
    c.multifluid.jwl_rho0  = pick<double>(mf["jwl_rho0"],  c.multifluid.jwl_rho0);
    c.multifluid.jwl_E0    = pick<double>(mf["jwl_E0"],    c.multifluid.jwl_E0);
    c.multifluid.rho_cj    = pick<double>(mf["rho_cj"],    c.multifluid.rho_cj);
    c.multifluid.p_cj      = pick<double>(mf["p_cj"],      c.multifluid.p_cj);
    c.multifluid.p_a_jwl   = pick<double>(mf["p_a"],       c.multifluid.p_a_jwl);
    c.multifluid.phi_switch = pick<double>(mf["phi_switch"], c.multifluid.phi_switch);
    c.multifluid.rho_ref   = pick<double>(mf["rho_ref"],   c.multifluid.rho_ref);
    c.multifluid.p_ref     = pick<double>(mf["p_ref"],     c.multifluid.p_ref);

    // Five-equation per-phase + IC parameters (eos = "five_equation").
    c.multifluid.ph1_kind  = pick<std::string>(mf["ph1_kind"], c.multifluid.ph1_kind);
    c.multifluid.ph2_kind  = pick<std::string>(mf["ph2_kind"], c.multifluid.ph2_kind);
    c.multifluid.ph1_gamma = pick<double>(mf["ph1_gamma"], c.multifluid.ph1_gamma);
    c.multifluid.ph1_pinf  = pick<double>(mf["ph1_pinf"],  c.multifluid.ph1_pinf);
    c.multifluid.ph2_gamma = pick<double>(mf["ph2_gamma"], c.multifluid.ph2_gamma);
    c.multifluid.ph2_pinf  = pick<double>(mf["ph2_pinf"],  c.multifluid.ph2_pinf);
    c.multifluid.fe_a1_in  = pick<double>(mf["a1_in"],     c.multifluid.fe_a1_in);
    c.multifluid.fe_a1_out = pick<double>(mf["a1_out"],    c.multifluid.fe_a1_out);
    c.multifluid.fe_rho1   = pick<double>(mf["rho1"],      c.multifluid.fe_rho1);
    c.multifluid.fe_rho2   = pick<double>(mf["rho2"],      c.multifluid.fe_rho2);
    c.multifluid.fe_p_in   = pick<double>(mf["p_in"],      c.multifluid.fe_p_in);
    c.multifluid.fe_p_out  = pick<double>(mf["p_out"],     c.multifluid.fe_p_out);
    c.multifluid.fe_u0     = pick<double>(mf["u0"],        c.multifluid.fe_u0);
    c.multifluid.fe_v0     = pick<double>(mf["v0"],        c.multifluid.fe_v0);
    c.multifluid.fe_w0     = pick<double>(mf["w0"],        c.multifluid.fe_w0);
    c.multifluid.ph1_jwl_A = pick<double>(mf["ph1_jwl_A"], c.multifluid.ph1_jwl_A);
    c.multifluid.ph1_jwl_B = pick<double>(mf["ph1_jwl_B"], c.multifluid.ph1_jwl_B);
    c.multifluid.ph1_jwl_R1 = pick<double>(mf["ph1_jwl_R1"], c.multifluid.ph1_jwl_R1);
    c.multifluid.ph1_jwl_R2 = pick<double>(mf["ph1_jwl_R2"], c.multifluid.ph1_jwl_R2);
    c.multifluid.ph1_jwl_omega = pick<double>(mf["ph1_jwl_omega"], c.multifluid.ph1_jwl_omega);
    c.multifluid.ph1_jwl_rho0 = pick<double>(mf["ph1_jwl_rho0"], c.multifluid.ph1_jwl_rho0);
    c.multifluid.ph2_jwl_A = pick<double>(mf["ph2_jwl_A"], c.multifluid.ph2_jwl_A);
    c.multifluid.ph2_jwl_B = pick<double>(mf["ph2_jwl_B"], c.multifluid.ph2_jwl_B);
    c.multifluid.ph2_jwl_R1 = pick<double>(mf["ph2_jwl_R1"], c.multifluid.ph2_jwl_R1);
    c.multifluid.ph2_jwl_R2 = pick<double>(mf["ph2_jwl_R2"], c.multifluid.ph2_jwl_R2);
    c.multifluid.ph2_jwl_omega = pick<double>(mf["ph2_jwl_omega"], c.multifluid.ph2_jwl_omega);
    c.multifluid.ph2_jwl_rho0 = pick<double>(mf["ph2_jwl_rho0"], c.multifluid.ph2_jwl_rho0);

    auto o = tbl["output"];
    c.output.out_dir          = pick<std::string>(o["out_dir"], c.output.out_dir);
    c.output.snapshot_every   = pick<int64_t>(o["snapshot_every"], c.output.snapshot_every);
    c.output.snapshot_dt      = pick<double>(o["snapshot_dt"],     c.output.snapshot_dt);
    c.output.stats_every      = pick<int64_t>(o["stats_every"],    c.output.stats_every);
    c.output.stats_dt         = pick<double>(o["stats_dt"],        c.output.stats_dt);
    c.output.spectra_dt       = pick<double>(o["spectra_dt"],      c.output.spectra_dt);
    c.output.spectra_every    = pick<int64_t>(o["spectra_every"],  c.output.spectra_every);
    c.output.checkpoint_every = pick<int64_t>(o["checkpoint_every"], c.output.checkpoint_every);
    c.output.checkpoint_dt    = pick<double>(o["checkpoint_dt"],   c.output.checkpoint_dt);
    c.output.write_spectra    = pick<bool>(o["write_spectra"],   c.output.write_spectra);
    c.output.write_helmholtz  = pick<bool>(o["write_helmholtz"], c.output.write_helmholtz);
    c.output.restart_path     = pick<std::string>(o["restart_path"], c.output.restart_path);

    return c;
}

}  // namespace blast
