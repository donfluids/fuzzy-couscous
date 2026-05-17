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

ICType parse_ic(std::string_view s) {
    if (s == "sod_x")          return ICType::SodX;
    if (s == "shu_osher_x")    return ICType::ShuOsherX;
    if (s == "sedov")          return ICType::Sedov;
    if (s == "taylor_green")   return ICType::TaylorGreen;
    if (s == "cbc")            return ICType::CBC;
    if (s == "tophat_sphere")  return ICType::TopHatSphere;
    if (s == "smooth_sphere")  return ICType::SmoothSphere;
    if (s == "cj_detonation")  return ICType::CJDetonation;
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
    c.physics.hyper_coeff = pick<double>(ph["hyper_coeff"], c.physics.hyper_coeff);

    auto a = tbl["afp"];
    c.afp.enabled = pick<bool>(a["enabled"], c.afp.enabled);
    c.afp.r_order = pick<int64_t>(a["r_order"], c.afp.r_order);
    c.afp.C_mu    = pick<double>(a["C_mu"], c.afp.C_mu);
    c.afp.C_beta  = pick<double>(a["C_beta"], c.afp.C_beta);
    c.afp.C_kappa = pick<double>(a["C_kappa"], c.afp.C_kappa);

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

    auto o = tbl["output"];
    c.output.out_dir          = pick<std::string>(o["out_dir"], c.output.out_dir);
    c.output.snapshot_every   = pick<int64_t>(o["snapshot_every"], c.output.snapshot_every);
    c.output.stats_every      = pick<int64_t>(o["stats_every"],    c.output.stats_every);
    c.output.checkpoint_every = pick<int64_t>(o["checkpoint_every"], c.output.checkpoint_every);
    c.output.write_spectra    = pick<bool>(o["write_spectra"],   c.output.write_spectra);
    c.output.write_helmholtz  = pick<bool>(o["write_helmholtz"], c.output.write_helmholtz);
    c.output.restart_path     = pick<std::string>(o["restart_path"], c.output.restart_path);

    return c;
}

}  // namespace blast
