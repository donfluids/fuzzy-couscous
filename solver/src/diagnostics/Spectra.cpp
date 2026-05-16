#include "diagnostics/Spectra.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <omp.h>

namespace blast {

namespace {

// Copy interior u_i = (rho u_i) / rho into a contiguous nx*ny*nz buffer.
void pack_velocity_component(const State& U, int comp, std::vector<Real>& out) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz();
    out.resize(static_cast<std::size_t>(nx) * ny * nz);
    const auto& rho = U[RHO];
    const auto& m   = U[RHOU + comp];

#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
#pragma omp simd
            for (int i = 0; i < nx; ++i) {
                const std::size_t idx =
                    static_cast<std::size_t>(i)
                    + nx * (static_cast<std::size_t>(j) + ny * k);
                out[idx] = m(i, j, k) / rho(i, j, k);
            }
}

// Wavenumber along the n-th dim for FFTW r2c output index (size n bins from
// 0..n-1 for non-Hermitian dims, 0..n/2 for the last dim). Wraps as integer
// frequency 0,1,..,n/2,-n/2+1,..,-1 for full dims and 0,1,..,n/2 for the
// reduced dim. Returns the discrete integer frequency.
int dft_freq(int idx, int N) {
    if (idx <= N / 2) return idx;
    return idx - N;
}

}  // namespace

ShellSpectrum velocity_spectrum(const State& U, const Grid& g, FFT3DPlan& plan) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz();
    const std::size_t N3 = static_cast<std::size_t>(nx) * ny * nz;
    const std::size_t Nc = plan.complex_size();
    const Real total_cells = static_cast<Real>(N3);

    std::vector<Real> field(N3);
    std::vector<std::complex<Real>> spec(Nc);

    // FFT each component, accumulate (1/2)|u_hat|^2 into shell bins.
    // Bin width is the smallest fundamental wavenumber so bin index = round(|k|).
    const Real k_fund_x = 2.0 * M_PI / g.lx;
    const Real k_fund_y = 2.0 * M_PI / g.ly;
    const Real k_fund_z = 2.0 * M_PI / g.lz;
    const Real k_fund   = std::min({k_fund_x, k_fund_y, k_fund_z});

    const int kmax = std::max({nx, ny, nz}) / 2;
    std::vector<Real> bin_sum(kmax + 1, 0.0);
    std::vector<long> bin_cnt(kmax + 1, 0);

    for (int comp = 0; comp < 3; ++comp) {
        pack_velocity_component(U, comp, field);
        plan.forward(field.data(), spec.data());

        // Index layout from fftw_plan_dft_r2c_3d(nz, ny, nx, ...):
        // outermost dim is the "z-like" of length nz, then ny, then the
        // reduced nx-like of length nx/2+1. But we labelled FFTW dims as
        // (nz, ny, nx), so the reduced dim is nx -> nx/2+1, and the storage
        // strides are (ny*(nx/2+1), nx/2+1, 1).
        const int nx_c = nx / 2 + 1;
#pragma omp parallel for collapse(2) schedule(static)
        for (int kz = 0; kz < nz; ++kz)
            for (int ky = 0; ky < ny; ++ky)
                for (int kx = 0; kx < nx_c; ++kx) {
                    const int fz = dft_freq(kz, nz);
                    const int fy = dft_freq(ky, ny);
                    const int fx = kx;          // r2c reduced dim: 0..nx/2

                    const std::size_t idx =
                        static_cast<std::size_t>(kx)
                        + nx_c * (static_cast<std::size_t>(ky) + ny * kz);
                    const std::complex<Real> uh = spec[idx];

                    // r2c packing: modes with fx != 0 and fx != nx/2 appear
                    // only once in the half-spectrum but represent both
                    // (kx,ky,kz) and (-kx,-ky,-kz). Double their contribution.
                    Real weight = (fx == 0 || fx == nx / 2) ? 1.0 : 2.0;

                    const Real kmag = std::sqrt(static_cast<Real>(fx * fx + fy * fy + fz * fz));
                    int b = static_cast<int>(std::round(kmag));
                    if (b > kmax) continue;
                    const Real e_mode = 0.5 * weight * (uh.real() * uh.real() + uh.imag() * uh.imag());
#pragma omp atomic
                    bin_sum[b] += e_mode;
                    if (comp == 0) {
#pragma omp atomic
                        bin_cnt[b] += 1;
                    }
                }
    }

    // Normalize: FFTW unnormalized forward gives sum_k |u_hat|^2 = N * sum_x |u|^2.
    // Energy density (1/2)<u^2> = (1/N) * sum_x (1/2) u^2 = (1/N^2) sum_k (1/2)|u_hat|^2.
    const Real norm = 1.0 / (total_cells * total_cells);

    ShellSpectrum sp;
    sp.k.resize(kmax + 1);
    sp.E.resize(kmax + 1);
    for (int b = 0; b <= kmax; ++b) {
        sp.k[b] = b * k_fund;
        sp.E[b] = bin_sum[b] * norm;
    }
    return sp;
}

HelmholtzResult helmholtz_decompose(const State& U, const Grid& g,
                                    FFT3DPlan& plan) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz();
    const std::size_t N3 = static_cast<std::size_t>(nx) * ny * nz;
    const std::size_t Nc = plan.complex_size();
    const Real total_cells = static_cast<Real>(N3);

    std::vector<Real> field(N3);
    std::vector<std::complex<Real>> uh(Nc), vh(Nc), wh(Nc);

    pack_velocity_component(U, 0, field); plan.forward(field.data(), uh.data());
    pack_velocity_component(U, 1, field); plan.forward(field.data(), vh.data());
    pack_velocity_component(U, 2, field); plan.forward(field.data(), wh.data());

    const int kmax = std::max({nx, ny, nz}) / 2;
    std::vector<Real> sol_sum(kmax + 1, 0.0), dil_sum(kmax + 1, 0.0);

    const int nx_c = nx / 2 + 1;
    const Real two_pi_Lx = 2.0 * M_PI / g.lx;
    const Real two_pi_Ly = 2.0 * M_PI / g.ly;
    const Real two_pi_Lz = 2.0 * M_PI / g.lz;
    const Real k_fund    = std::min({two_pi_Lx, two_pi_Ly, two_pi_Lz});

#pragma omp parallel for collapse(2) schedule(static)
    for (int kz = 0; kz < nz; ++kz)
        for (int ky = 0; ky < ny; ++ky)
            for (int kx = 0; kx < nx_c; ++kx) {
                const int fz = dft_freq(kz, nz);
                const int fy = dft_freq(ky, ny);
                const int fx = kx;

                if (fx == 0 && fy == 0 && fz == 0) continue;   // mean

                const Real kxp = fx * two_pi_Lx;
                const Real kyp = fy * two_pi_Ly;
                const Real kzp = fz * two_pi_Lz;
                const Real kmag2 = kxp * kxp + kyp * kyp + kzp * kzp;
                if (kmag2 == 0) continue;
                const Real inv_kmag2 = 1.0 / kmag2;

                const std::size_t idx =
                    static_cast<std::size_t>(kx)
                    + nx_c * (static_cast<std::size_t>(ky) + ny * kz);
                const std::complex<Real> ux = uh[idx];
                const std::complex<Real> uy = vh[idx];
                const std::complex<Real> uz = wh[idx];

                const std::complex<Real> k_dot_u =
                    kxp * ux + kyp * uy + kzp * uz;

                const std::complex<Real> dil_x = k_dot_u * kxp * inv_kmag2;
                const std::complex<Real> dil_y = k_dot_u * kyp * inv_kmag2;
                const std::complex<Real> dil_z = k_dot_u * kzp * inv_kmag2;
                const std::complex<Real> sol_x = ux - dil_x;
                const std::complex<Real> sol_y = uy - dil_y;
                const std::complex<Real> sol_z = uz - dil_z;

                Real weight = (fx == 0 || fx == nx / 2) ? 1.0 : 2.0;
                const Real e_sol = 0.5 * weight * (
                    sol_x.real() * sol_x.real() + sol_x.imag() * sol_x.imag()
                  + sol_y.real() * sol_y.real() + sol_y.imag() * sol_y.imag()
                  + sol_z.real() * sol_z.real() + sol_z.imag() * sol_z.imag());
                const Real e_dil = 0.5 * weight * (
                    dil_x.real() * dil_x.real() + dil_x.imag() * dil_x.imag()
                  + dil_y.real() * dil_y.real() + dil_y.imag() * dil_y.imag()
                  + dil_z.real() * dil_z.real() + dil_z.imag() * dil_z.imag());

                const Real kmag_int = std::sqrt(static_cast<Real>(fx * fx + fy * fy + fz * fz));
                int b = static_cast<int>(std::round(kmag_int));
                if (b > kmax) continue;
#pragma omp atomic
                sol_sum[b] += e_sol;
#pragma omp atomic
                dil_sum[b] += e_dil;
            }

    const Real norm = 1.0 / (total_cells * total_cells);
    HelmholtzResult r;
    r.E_sol.k.resize(kmax + 1); r.E_sol.E.resize(kmax + 1);
    r.E_dil.k.resize(kmax + 1); r.E_dil.E.resize(kmax + 1);
    Real K_sol = 0, K_dil = 0;
    for (int b = 0; b <= kmax; ++b) {
        r.E_sol.k[b] = b * k_fund;
        r.E_dil.k[b] = b * k_fund;
        r.E_sol.E[b] = sol_sum[b] * norm;
        r.E_dil.E[b] = dil_sum[b] * norm;
        K_sol += r.E_sol.E[b];
        K_dil += r.E_dil.E[b];
    }
    r.K_sol = K_sol;
    r.K_dil = K_dil;
    return r;
}

}  // namespace blast
