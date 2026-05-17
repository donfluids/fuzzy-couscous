#include "ic/Canonical.hpp"

#include <array>
#include <cmath>
#include <random>

namespace blast {

void set_from_primitive(State& U, int i, int j, int k, const IdealGas& eos,
                        Real rho, Real u, Real v, Real w, Real p) {
    const Real e_int = p / (eos.eos.gamma - 1.0);   // internal energy density
    const Real ke    = 0.5 * rho * (u*u + v*v + w*w);
    U[RHO ](i,j,k) = rho;
    U[RHOU](i,j,k) = rho * u;
    U[RHOV](i,j,k) = rho * v;
    U[RHOW](i,j,k) = rho * w;
    U[RHOE](i,j,k) = e_int + ke;
}

void ic_sod_x(State& U, const Grid& g, const IdealGas& eos) {
    const Real x_mid = g.x0 + 0.5 * g.lx;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real x = g.xc(i);
                const bool left = (x < x_mid);
                const Real rho = left ? 1.0   : 0.125;
                const Real p   = left ? 1.0   : 0.1;
                set_from_primitive(U, i, j, k, eos, rho, 0.0, 0.0, 0.0, p);
            }
}

void ic_shu_osher_x(State& U, const Grid& g, const IdealGas& eos) {
    // Domain [-5, 5]; shock at x = -4.
    const Real x_shock = g.x0 + 0.1 * g.lx;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real x = g.xc(i);
                Real rho, u, p;
                if (x < x_shock) {
                    rho = 3.857143;
                    u   = 2.629369;
                    p   = 10.33333;
                } else {
                    rho = 1.0 + 0.2 * std::sin(5.0 * x);
                    u   = 0.0;
                    p   = 1.0;
                }
                set_from_primitive(U, i, j, k, eos, rho, u, 0.0, 0.0, p);
            }
}

void ic_sedov_3d(State& U, const Grid& g, const IdealGas& eos,
                 Real E_total, Real rho_ambient, Real p_ambient,
                 Real r_blast) {
    const Real xc = g.x0 + 0.5 * g.lx;
    const Real yc = g.y0 + 0.5 * g.ly;
    const Real zc = g.z0 + 0.5 * g.lz;

    const Real V_blast = (4.0 / 3.0) * M_PI * r_blast * r_blast * r_blast;
    const Real p_blast = (eos.eos.gamma - 1.0) * E_total / V_blast;

    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real xv = g.xc(i) - xc;
                const Real yv = g.yc(j) - yc;
                const Real zv = g.zc(k) - zc;
                const Real r  = std::sqrt(xv*xv + yv*yv + zv*zv);
                const Real p  = (r < r_blast) ? p_blast : p_ambient;
                set_from_primitive(U, i, j, k, eos, rho_ambient, 0.0, 0.0, 0.0, p);
            }
}

void ic_taylor_green_3d(State& U, const Grid& g, const IdealGas& eos,
                        Real V0, Real rho_0, Real M_0) {
    const Real gamma = eos.eos.gamma;
    const Real p_ref = rho_0 * V0 * V0 / (gamma * M_0 * M_0);

    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real x = g.xc(i), y = g.yc(j), z = g.zc(k);
                const Real u =  V0 * std::sin(x) * std::cos(y) * std::cos(z);
                const Real v = -V0 * std::cos(x) * std::sin(y) * std::cos(z);
                const Real w =  0.0;
                const Real p = p_ref + (rho_0 * V0 * V0 / 16.0)
                              * (std::cos(2.0 * x) + std::cos(2.0 * y))
                              * (std::cos(2.0 * z) + 2.0);
                set_from_primitive(U, i, j, k, eos, rho_0, u, v, w, p);
            }
}

// Real spherical harmonic Y_{4,2}(theta, phi) up to normalization.
// theta polar from +z, phi azimuth around z. In Cartesian:
//   Y_{4,2} ~ (3 sqrt(5) / (8 sqrt(pi))) * (x^2 - y^2) (7 z^2 - r^2) / r^4
// The amplitude doesn't matter for our perturbation use; we scale it back.
static inline Real y42_unit(Real x, Real y, Real z) {
    const Real r2 = x * x + y * y + z * z;
    if (r2 < 1e-30) return 0.0;
    return (x * x - y * y) * (7.0 * z * z - r2) / (r2 * r2);
}

// Ensemble random angular perturbation: low-l spherical harmonics summed with
// seeded Gaussian coefficients. Compact in physical space, well-resolved by
// the angular grid, and different seeds give statistically distinct ICs.
struct AngularPerturbation {
    std::array<std::array<Real, 13>, 7> c{};   // c[l][m+6] for l=0..6
    Real overall_amp = 0.0;

    void seed(int s, Real amp) {
        overall_amp = amp;
        if (amp <= 0.0) return;
        std::mt19937 rng(static_cast<unsigned int>(s));
        std::normal_distribution<Real> nd(0.0, 1.0);
        for (int l = 2; l <= 6; ++l)
            for (int m = -l; m <= l; ++m) c[l][m + 6] = nd(rng);
    }

    // Evaluate sum_{l,m} c_{l,m} Y_{l,m}(theta, phi) up to l = 6, using the
    // Cartesian polynomial forms (unnormalized) for l = 2..6.
    Real eval(Real x, Real y, Real z) const {
        if (overall_amp <= 0.0) return 0.0;
        const Real r2 = x*x + y*y + z*z;
        if (r2 < 1e-30) return 0.0;
        const Real r4 = r2 * r2;
        const Real r6 = r4 * r2;
        Real s = 0.0;
        // l = 2 (5 modes), unnormalized:
        s += c[2][-2 + 6] * (2.0 * x * y) / r2;
        s += c[2][-1 + 6] * (2.0 * y * z) / r2;
        s += c[2][ 0 + 6] * (3.0 * z * z - r2) / r2;
        s += c[2][ 1 + 6] * (2.0 * x * z) / r2;
        s += c[2][ 2 + 6] * (x * x - y * y) / r2;
        // l = 4, just our Y_{4,2}-like family for variety:
        s += c[4][-2 + 6] * (x * y * (7.0 * z * z - r2)) / r4;
        s += c[4][ 2 + 6] * ((x * x - y * y) * (7.0 * z * z - r2)) / r4;
        s += c[4][ 0 + 6] * (35.0 * z*z*z*z - 30.0 * z*z * r2 + 3.0 * r4) / r4;
        // l = 6 octupolar contribution adds finer angular structure:
        s += c[6][ 0 + 6] * (231.0 * z*z*z*z*z*z - 315.0 * z*z*z*z * r2
                            + 105.0 * z*z * r4 - 5.0 * r6) / r6;
        return overall_amp * s;
    }
};

void ic_sphere_blast_3d(State& U, const Grid& g, const IdealGas& eos,
                        Real rho_blast, Real T_blast,
                        Real rho_ambient, Real T_ambient,
                        Real r_blast, Real tanh_thickness, Real Y42_amp,
                        Real ensemble_amp, int ensemble_seed,
                        Real x_center, Real y_center, Real z_center) {
    const Real xc = std::isnan(x_center) ? g.x0 + 0.5 * g.lx : x_center;
    const Real yc = std::isnan(y_center) ? g.y0 + 0.5 * g.ly : y_center;
    const Real zc = std::isnan(z_center) ? g.z0 + 0.5 * g.lz : z_center;
    const Real p_blast   = rho_blast   * eos.eos.R * T_blast;
    const Real p_ambient = rho_ambient * eos.eos.R * T_ambient;
    const Real safe_thickness = std::max(tanh_thickness, 1e-12);

    AngularPerturbation pert;
    pert.seed(ensemble_seed, ensemble_amp);

    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real x = g.xc(i) - xc;
                const Real y = g.yc(j) - yc;
                const Real z = g.zc(k) - zc;
                const Real r = std::sqrt(x*x + y*y + z*z);
                Real r_eff = r_blast * (1.0
                                         + Y42_amp * y42_unit(x, y, z)
                                         + pert.eval(x, y, z));
                Real w;
                if (tanh_thickness > 0.0) {
                    w = 0.5 * (1.0 - std::tanh((r - r_eff) / safe_thickness));
                } else {
                    w = (r < r_eff) ? 1.0 : 0.0;
                }
                const Real rho = rho_ambient + w * (rho_blast - rho_ambient);
                const Real p   = p_ambient   + w * (p_blast   - p_ambient);
                set_from_primitive(U, i, j, k, eos, rho, 0.0, 0.0, 0.0, p);
            }
}

void ic_cj_detonation_3d(State& U, const Grid& g, const IdealGas& eos,
                         Real rho_0, Real T_0, Real q_specific,
                         Real r_cj, Real tanh_thickness, Real Y42_amp,
                         Real x_center, Real y_center, Real z_center) {
    const Real gamma = eos.eos.gamma;
    const Real R     = eos.eos.R;
    const Real p_0   = rho_0 * R * T_0;
    const Real c_0   = std::sqrt(gamma * R * T_0);

    // Exact ideal-gas CJ detonation Mach number (Williams, Combustion Theory
    // 1985, eq. 6.28). Define alpha = (gamma + 1) q / c_0^2; then
    //   M_D^2 = 1 + alpha + sqrt(alpha^2 + 2 alpha)
    // recovers both M_D -> 1 as q -> 0 and M_D^2 -> 2 (gamma+1) q / c_0^2
    // for q >> c_0^2 (strong detonation).
    const Real q = q_specific;                                 // J/kg
    const Real alpha = (gamma + 1.0) * q / (c_0 * c_0);
    const Real M_D2  = 1.0 + alpha + std::sqrt(alpha * alpha + 2.0 * alpha);
    const Real D     = c_0 * std::sqrt(M_D2);

    // Post-CJ-shock state in the lab frame.
    const Real p_cj   = p_0   * (1.0 + gamma * M_D2) / (gamma + 1.0);
    const Real rho_cj = rho_0 * (gamma + 1.0) * M_D2 / (gamma * M_D2 + 1.0);
    const Real u_cj   = D * (1.0 - rho_0 / rho_cj);            // radial outward

    const Real xc = std::isnan(x_center) ? g.x0 + 0.5 * g.lx : x_center;
    const Real yc = std::isnan(y_center) ? g.y0 + 0.5 * g.ly : y_center;
    const Real zc = std::isnan(z_center) ? g.z0 + 0.5 * g.lz : z_center;
    const Real safe_thickness = std::max(tanh_thickness, 1e-12);

    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real x = g.xc(i) - xc;
                const Real y = g.yc(j) - yc;
                const Real z = g.zc(k) - zc;
                const Real r = std::sqrt(x*x + y*y + z*z);
                Real r_eff = r_cj * (1.0 + Y42_amp * y42_unit(x, y, z));
                Real w;
                if (tanh_thickness > 0.0) {
                    w = 0.5 * (1.0 - std::tanh((r - r_eff) / safe_thickness));
                } else {
                    w = (r < r_eff) ? 1.0 : 0.0;
                }
                const Real rho = rho_0 + w * (rho_cj - rho_0);
                const Real p   = p_0   + w * (p_cj   - p_0);
                Real ur = 0.0;
                if (w > 0.0 && r > 1e-12) ur = w * u_cj;
                const Real inv_r = (r > 1e-12) ? 1.0 / r : 0.0;
                const Real u = ur * x * inv_r;
                const Real v = ur * y * inv_r;
                const Real wz = ur * z * inv_r;
                set_from_primitive(U, i, j, k, eos, rho, u, v, wz, p);
            }
}

void ic_entropy_wave_3d(State& U, const Grid& g, const IdealGas& eos,
                        Real amplitude, Real u0, Real v0, Real w0,
                        Real rho_0, Real p_0) {
    const Real kx = 2.0 * M_PI / g.lx;
    const Real ky = 2.0 * M_PI / g.ly;
    const Real kz = 2.0 * M_PI / g.lz;
    for (int k = -U.ng(); k < g.nz + U.ng(); ++k)
        for (int j = -U.ng(); j < g.ny + U.ng(); ++j)
            for (int i = -U.ng(); i < g.nx + U.ng(); ++i) {
                const Real x = g.xc(i), y = g.yc(j), z = g.zc(k);
                const Real rho = rho_0 * (1.0
                    + amplitude * std::sin(kx * x) * std::sin(ky * y) * std::sin(kz * z));
                set_from_primitive(U, i, j, k, eos, rho, u0, v0, w0, p_0);
            }
}

void ic_isentropic_vortex(State& U, const Grid& g, const IdealGas& eos,
                          Real eps, Real u_inf, Real v_inf,
                          Real x_c, Real y_c) {
    const Real gamma = eos.eos.gamma;
    const Real gm1   = gamma - 1.0;
    const Real coeff = gm1 * eps * eps / (8.0 * gamma * M_PI * M_PI);
    const Real two_pi_inv = 1.0 / (2.0 * M_PI);

    for (int k = -U.ng(); k < g.nz + U.ng(); ++k)
        for (int j = -U.ng(); j < g.ny + U.ng(); ++j)
            for (int i = -U.ng(); i < g.nx + U.ng(); ++i) {
                const Real x = g.xc(i), y = g.yc(j);
                const Real xd = x - x_c;
                const Real yd = y - y_c;
                const Real r2 = xd * xd + yd * yd;
                const Real e_factor = std::exp((1.0 - r2) * 0.5);
                const Real u =  u_inf - eps * yd * two_pi_inv * e_factor;
                const Real v =  v_inf + eps * xd * two_pi_inv * e_factor;
                const Real T = 1.0 - coeff * std::exp(1.0 - r2);
                const Real rho = std::pow(T, 1.0 / gm1);
                const Real p   = std::pow(T, gamma / gm1);
                set_from_primitive(U, i, j, k, eos, rho, u, v, 0.0, p);
            }
}

void ic_density_wave_x(State& U, const Grid& g, const IdealGas& eos,
                       Real amplitude, Real kwave, Real u0) {
    const Real k_phys = 2.0 * M_PI * kwave / g.lx;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real x = g.xc(i);
                const Real rho = 1.0 + amplitude * std::sin(k_phys * x);
                set_from_primitive(U, i, j, k, eos, rho, u0, 0.0, 0.0, 1.0);
            }
}

}  // namespace blast
