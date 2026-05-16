#include "ic/Canonical.hpp"

#include <cmath>

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
