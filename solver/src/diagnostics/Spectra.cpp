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

#ifdef BLAST_MPI

namespace {

// Pack rank's local velocity component into a contiguous flat buffer
// (interior cells only, i-fastest order, no ghosts).
void pack_velocity_component_local(const State& U, int comp,
                                   std::vector<double>& out) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz();
    out.resize(static_cast<std::size_t>(nx) * ny * nz);
    const auto& rho = U[RHO];
    const auto& m   = U[RHOU + comp];
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const std::size_t idx = static_cast<std::size_t>(i)
                    + nx * (static_cast<std::size_t>(j) + ny * k);
                out[idx] = m(i, j, k) / rho(i, j, k);
            }
}

// Gather flat local buffer to rank 0 at the right hyperslab in `global_buf`.
// Tags 1000-1002 are used.
void gather_to_rank0(const std::vector<double>& local_buf,
                     const Grid& global_g, const Domain& d,
                     std::vector<double>& global_buf,
                     int nx_l, int ny_l, int nz_l) {
    const int rank = d.rank();
    const int size = d.size();
    const auto ext = d.global_extent();
    const int nx_g = ext[0], ny_g = ext[1], nz_g = ext[2];

    if (rank == 0) {
        global_buf.resize(static_cast<std::size_t>(nx_g) * ny_g * nz_g);

        // Place rank-0 own data first.
        const auto off0 = d.global_offset(global_g);
        for (int k = 0; k < nz_l; ++k)
            for (int j = 0; j < ny_l; ++j)
                for (int i = 0; i < nx_l; ++i) {
                    const std::size_t li = static_cast<std::size_t>(i)
                        + nx_l * (static_cast<std::size_t>(j) + ny_l * k);
                    const std::size_t gi = static_cast<std::size_t>(i + off0[0])
                        + nx_g * (static_cast<std::size_t>(j + off0[1])
                                  + ny_g * (k + off0[2]));
                    global_buf[gi] = local_buf[li];
                }

        // Receive from other ranks.
        for (int r = 1; r < size; ++r) {
            int meta[6];   // off[3], ext[3]
            MPI_Recv(meta, 6, MPI_INT, r, 1000, d.comm(), MPI_STATUS_IGNORE);
            const int off_x = meta[0], off_y = meta[1], off_z = meta[2];
            const int ex = meta[3], ey = meta[4], ez = meta[5];
            const std::size_t rsize = static_cast<std::size_t>(ex) * ey * ez;
            std::vector<double> rbuf(rsize);
            MPI_Recv(rbuf.data(), static_cast<int>(rsize), MPI_DOUBLE, r,
                     1002, d.comm(), MPI_STATUS_IGNORE);
            for (int k = 0; k < ez; ++k)
                for (int j = 0; j < ey; ++j)
                    for (int i = 0; i < ex; ++i) {
                        const std::size_t li = static_cast<std::size_t>(i)
                            + ex * (static_cast<std::size_t>(j) + ey * k);
                        const std::size_t gi = static_cast<std::size_t>(i + off_x)
                            + nx_g * (static_cast<std::size_t>(j + off_y)
                                      + ny_g * (k + off_z));
                        global_buf[gi] = rbuf[li];
                    }
        }
    } else {
        const auto off = d.global_offset(global_g);
        int meta[6] = {static_cast<int>(off[0]), static_cast<int>(off[1]),
                       static_cast<int>(off[2]), nx_l, ny_l, nz_l};
        MPI_Send(meta, 6, MPI_INT, 0, 1000, d.comm());
        MPI_Send(local_buf.data(), static_cast<int>(local_buf.size()),
                 MPI_DOUBLE, 0, 1002, d.comm());
    }
}

}  // namespace

ShellSpectrum velocity_spectrum_mpi(const State& U, const Grid& global_g,
                                    FFT3DPlan& plan, const Domain& d) {
    const int nx_l = U.nx(), ny_l = U.ny(), nz_l = U.nz();
    const auto ext = d.global_extent();
    const int nx_g = ext[0], ny_g = ext[1], nz_g = ext[2];

    std::vector<double> local_buf, global_buf;
    std::vector<std::complex<Real>> spec(plan.complex_size());

    const Real k_fund_x = 2.0 * M_PI / global_g.lx;
    const Real k_fund_y = 2.0 * M_PI / global_g.ly;
    const Real k_fund_z = 2.0 * M_PI / global_g.lz;
    const Real k_fund   = std::min({k_fund_x, k_fund_y, k_fund_z});

    const int kmax = std::max({nx_g, ny_g, nz_g}) / 2;
    std::vector<Real> bin_sum(kmax + 1, 0.0);

    for (int comp = 0; comp < 3; ++comp) {
        pack_velocity_component_local(U, comp, local_buf);
        gather_to_rank0(local_buf, global_g, d, global_buf, nx_l, ny_l, nz_l);

        if (d.rank() != 0) continue;
        plan.forward(global_buf.data(), spec.data());

        const int nx_c = nx_g / 2 + 1;
        for (int kz = 0; kz < nz_g; ++kz)
            for (int ky = 0; ky < ny_g; ++ky)
                for (int kx = 0; kx < nx_c; ++kx) {
                    const int fz = (kz <= nz_g / 2) ? kz : kz - nz_g;
                    const int fy = (ky <= ny_g / 2) ? ky : ky - ny_g;
                    const int fx = kx;
                    const std::size_t idx = static_cast<std::size_t>(kx)
                        + nx_c * (static_cast<std::size_t>(ky) + ny_g * kz);
                    const std::complex<Real> uh = spec[idx];
                    const Real weight = (fx == 0 || fx == nx_g / 2) ? 1.0 : 2.0;
                    const Real kmag = std::sqrt(static_cast<Real>(
                        fx * fx + fy * fy + fz * fz));
                    int b = static_cast<int>(std::round(kmag));
                    if (b > kmax) continue;
                    bin_sum[b] += 0.5 * weight *
                        (uh.real() * uh.real() + uh.imag() * uh.imag());
                }
    }

    ShellSpectrum sp;
    if (d.rank() == 0) {
        const Real total_cells = static_cast<Real>(nx_g) * ny_g * nz_g;
        const Real norm = 1.0 / (total_cells * total_cells);
        sp.k.resize(kmax + 1);
        sp.E.resize(kmax + 1);
        for (int b = 0; b <= kmax; ++b) {
            sp.k[b] = b * k_fund;
            sp.E[b] = bin_sum[b] * norm;
        }
    }
    return sp;
}

HelmholtzResult helmholtz_decompose_mpi(const State& U, const Grid& global_g,
                                        FFT3DPlan& plan, const Domain& d) {
    const int nx_l = U.nx(), ny_l = U.ny(), nz_l = U.nz();
    const auto ext = d.global_extent();
    const int nx_g = ext[0], ny_g = ext[1], nz_g = ext[2];

    std::vector<double> local_buf, gu, gv, gw;
    pack_velocity_component_local(U, 0, local_buf);
    gather_to_rank0(local_buf, global_g, d, gu, nx_l, ny_l, nz_l);
    pack_velocity_component_local(U, 1, local_buf);
    gather_to_rank0(local_buf, global_g, d, gv, nx_l, ny_l, nz_l);
    pack_velocity_component_local(U, 2, local_buf);
    gather_to_rank0(local_buf, global_g, d, gw, nx_l, ny_l, nz_l);

    HelmholtzResult r;
    if (d.rank() != 0) return r;

    std::vector<std::complex<Real>> uh(plan.complex_size()), vh(plan.complex_size()),
                                    wh(plan.complex_size());
    plan.forward(gu.data(), uh.data());
    plan.forward(gv.data(), vh.data());
    plan.forward(gw.data(), wh.data());

    const int kmax = std::max({nx_g, ny_g, nz_g}) / 2;
    std::vector<Real> sol_sum(kmax + 1, 0.0), dil_sum(kmax + 1, 0.0);

    const int nx_c = nx_g / 2 + 1;
    const Real two_pi_Lx = 2.0 * M_PI / global_g.lx;
    const Real two_pi_Ly = 2.0 * M_PI / global_g.ly;
    const Real two_pi_Lz = 2.0 * M_PI / global_g.lz;
    const Real k_fund    = std::min({two_pi_Lx, two_pi_Ly, two_pi_Lz});

    for (int kz = 0; kz < nz_g; ++kz)
        for (int ky = 0; ky < ny_g; ++ky)
            for (int kx = 0; kx < nx_c; ++kx) {
                const int fz = (kz <= nz_g / 2) ? kz : kz - nz_g;
                const int fy = (ky <= ny_g / 2) ? ky : ky - ny_g;
                const int fx = kx;
                if (fx == 0 && fy == 0 && fz == 0) continue;
                const Real kxp = fx * two_pi_Lx;
                const Real kyp = fy * two_pi_Ly;
                const Real kzp = fz * two_pi_Lz;
                const Real kmag2 = kxp * kxp + kyp * kyp + kzp * kzp;
                if (kmag2 == 0) continue;
                const Real inv_kmag2 = 1.0 / kmag2;

                const std::size_t idx = static_cast<std::size_t>(kx)
                    + nx_c * (static_cast<std::size_t>(ky) + ny_g * kz);
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

                const Real weight = (fx == 0 || fx == nx_g / 2) ? 1.0 : 2.0;
                const Real e_sol = 0.5 * weight * (
                    sol_x.real() * sol_x.real() + sol_x.imag() * sol_x.imag()
                  + sol_y.real() * sol_y.real() + sol_y.imag() * sol_y.imag()
                  + sol_z.real() * sol_z.real() + sol_z.imag() * sol_z.imag());
                const Real e_dil = 0.5 * weight * (
                    dil_x.real() * dil_x.real() + dil_x.imag() * dil_x.imag()
                  + dil_y.real() * dil_y.real() + dil_y.imag() * dil_y.imag()
                  + dil_z.real() * dil_z.real() + dil_z.imag() * dil_z.imag());

                const Real kmag_int = std::sqrt(static_cast<Real>(
                    fx * fx + fy * fy + fz * fz));
                int b = static_cast<int>(std::round(kmag_int));
                if (b > kmax) continue;
                sol_sum[b] += e_sol;
                dil_sum[b] += e_dil;
            }

    const Real total_cells = static_cast<Real>(nx_g) * ny_g * nz_g;
    const Real norm = 1.0 / (total_cells * total_cells);
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

// ----------------------------------------------------------------------------
// v2: Distributed FFTW3-MPI slab decomposition. Each rank handles
// `plan.local_nz()` z-planes of the global field. Redistribution from the
// 3D Cartesian decomposition to the 1D z-slab layout uses MPI_Alltoallv.
// All ranks return the same populated result via MPI_Allreduce of the bins.
// ----------------------------------------------------------------------------

namespace {

struct CartDesc {
    int i_off, j_off, k_off;
    int nx, ny, nz;
};
struct SlabDesc {
    int k_start, k_count;
};

void gather_cart_descs(const Domain& d, const Grid& global,
                       std::vector<CartDesc>& out) {
    Grid lg = d.local_grid(global);
    auto off = d.global_offset(global);
    int my[6] = {
        static_cast<int>(off[0]), static_cast<int>(off[1]), static_cast<int>(off[2]),
        lg.nx, lg.ny, lg.nz};
    std::vector<int> buf(static_cast<std::size_t>(d.size()) * 6);
    MPI_Allgather(my, 6, MPI_INT, buf.data(), 6, MPI_INT, d.comm());
    out.resize(static_cast<std::size_t>(d.size()));
    for (int r = 0; r < d.size(); ++r) {
        out[r].i_off = buf[6*r + 0];
        out[r].j_off = buf[6*r + 1];
        out[r].k_off = buf[6*r + 2];
        out[r].nx    = buf[6*r + 3];
        out[r].ny    = buf[6*r + 4];
        out[r].nz    = buf[6*r + 5];
    }
}

void gather_slab_descs(const FFT3DPlanMPI& plan, int size,
                       std::vector<SlabDesc>& out) {
    int my[2] = {plan.local_z_start(), plan.local_nz()};
    std::vector<int> buf(static_cast<std::size_t>(size) * 2);
    MPI_Allgather(my, 2, MPI_INT, buf.data(), 2, MPI_INT, plan.comm());
    out.resize(static_cast<std::size_t>(size));
    for (int r = 0; r < size; ++r) {
        out[r].k_start = buf[2*r + 0];
        out[r].k_count = buf[2*r + 1];
    }
}

// Redistribute one velocity component from the 3D Cartesian layout
// (interior cells of U, packed via pack_velocity_component_local) into the
// FFTW slab buffer. The output respects the in-place r2c padding so the
// transform can be executed straight away.
void redistribute_cart_to_slab(const std::vector<double>& local_buf,
                               const std::vector<CartDesc>& cart,
                               const std::vector<SlabDesc>& slab,
                               FFT3DPlanMPI& plan,
                               const Domain& d) {
    const int size = d.size();
    const int my_rank = d.rank();
    const CartDesc& my = cart[my_rank];
    const SlabDesc& mys = slab[my_rank];
    const int nx_g_plan = plan.nx_global();
    const int ny_g_plan = plan.ny_global();
    const int row_stride   = plan.real_row_stride();
    const std::size_t plane_stride =
        static_cast<std::size_t>(ny_g_plan) * row_stride;

    std::vector<int> send_counts(size, 0), send_displs(size, 0);
    std::vector<int> recv_counts(size, 0), recv_displs(size, 0);
    long long total_send = 0, total_recv = 0;

    for (int r = 0; r < size; ++r) {
        const int sl = slab[r].k_start;
        const int sh = sl + slab[r].k_count;
        const int kk_lo = std::max(my.k_off, sl);
        const int kk_hi = std::min(my.k_off + my.nz, sh);
        const int kk_n  = std::max(0, kk_hi - kk_lo);
        send_counts[r] = my.nx * my.ny * kk_n;
        send_displs[r] = static_cast<int>(total_send);
        total_send += send_counts[r];
    }
    for (int s = 0; s < size; ++s) {
        const int kk_lo = std::max(cart[s].k_off, mys.k_start);
        const int kk_hi = std::min(cart[s].k_off + cart[s].nz,
                                   mys.k_start + mys.k_count);
        const int kk_n  = std::max(0, kk_hi - kk_lo);
        recv_counts[s] = cart[s].nx * cart[s].ny * kk_n;
        recv_displs[s] = static_cast<int>(total_recv);
        total_recv += recv_counts[s];
    }

    std::vector<double> send_buf(static_cast<std::size_t>(total_send));
    std::vector<double> recv_buf(static_cast<std::size_t>(total_recv));

    // Pack send buffer.
    for (int r = 0; r < size; ++r) {
        if (send_counts[r] == 0) continue;
        const int sl = slab[r].k_start;
        const int sh = sl + slab[r].k_count;
        const int kk_lo = std::max(my.k_off, sl);
        const int kk_hi = std::min(my.k_off + my.nz, sh);
        double* dst = send_buf.data() + send_displs[r];
        std::size_t p = 0;
        for (int k_g = kk_lo; k_g < kk_hi; ++k_g) {
            const int k_loc = k_g - my.k_off;
            for (int j_loc = 0; j_loc < my.ny; ++j_loc) {
                const std::size_t row_base = static_cast<std::size_t>(my.nx)
                    * (static_cast<std::size_t>(j_loc)
                       + static_cast<std::size_t>(my.ny) * k_loc);
                for (int i_loc = 0; i_loc < my.nx; ++i_loc) {
                    dst[p++] = local_buf[row_base + i_loc];
                }
            }
        }
    }

    MPI_Alltoallv(send_buf.data(), send_counts.data(), send_displs.data(),
                  MPI_DOUBLE,
                  recv_buf.data(), recv_counts.data(), recv_displs.data(),
                  MPI_DOUBLE,
                  d.comm());

    // Zero the padded i-row tail to avoid garbage in FFTW's reduced columns
    // (the columns i in [nx, 2*(nx/2+1)) are unused but must be defined for
    // in-place r2c).
    double* rb = plan.real_buf();
    const std::size_t slab_real_words =
        static_cast<std::size_t>(plan.local_nz()) * plane_stride;
    std::fill(rb, rb + slab_real_words, 0.0);

    // Unpack into FFTW slab buffer at global (i_g, j_g, k_g_local).
    for (int s = 0; s < size; ++s) {
        if (recv_counts[s] == 0) continue;
        const int kk_lo = std::max(cart[s].k_off, mys.k_start);
        const int kk_hi = std::min(cart[s].k_off + cart[s].nz,
                                   mys.k_start + mys.k_count);
        const double* src = recv_buf.data() + recv_displs[s];
        std::size_t p = 0;
        for (int k_g = kk_lo; k_g < kk_hi; ++k_g) {
            const int k_loc = k_g - mys.k_start;
            for (int j_loc = 0; j_loc < cart[s].ny; ++j_loc) {
                const int j_g = cart[s].j_off + j_loc;
                const std::size_t row_base = static_cast<std::size_t>(row_stride) * j_g
                    + plane_stride * k_loc;
                for (int i_loc = 0; i_loc < cart[s].nx; ++i_loc) {
                    const int i_g = cart[s].i_off + i_loc;
                    rb[row_base + i_g] = src[p++];
                }
            }
        }
    }
}

}  // namespace

ShellSpectrum velocity_spectrum_mpi_dist(const State& U, const Grid& global_g,
                                         FFT3DPlanMPI& plan, const Domain& d) {
    const int nx_g = plan.nx_global();
    const int ny_g = plan.ny_global();
    const int nz_g = plan.nz_global();

    std::vector<CartDesc> cart;
    std::vector<SlabDesc> slab;
    gather_cart_descs(d, global_g, cart);
    gather_slab_descs(plan, d.size(), slab);

    const Real k_fund_x = 2.0 * M_PI / global_g.lx;
    const Real k_fund_y = 2.0 * M_PI / global_g.ly;
    const Real k_fund_z = 2.0 * M_PI / global_g.lz;
    const Real k_fund   = std::min({k_fund_x, k_fund_y, k_fund_z});
    const int kmax = std::max({nx_g, ny_g, nz_g}) / 2;

    std::vector<Real> bin_sum_local(kmax + 1, 0.0);
    std::vector<double> local_buf;

    const int nx_c = nx_g / 2 + 1;
    const int my_kz0  = plan.local_z_start();
    const int my_nz_l = plan.local_nz();

    for (int comp = 0; comp < 3; ++comp) {
        pack_velocity_component_local(U, comp, local_buf);
        redistribute_cart_to_slab(local_buf, cart, slab, plan, d);
        plan.forward();

        const std::complex<Real>* spec_buf = plan.complex_buf();
        // Complex stride: (kx + nx_c * (ky + ny_g * kz_local)).
        for (int k_loc = 0; k_loc < my_nz_l; ++k_loc) {
            const int kz_g = my_kz0 + k_loc;
            const int fz = (kz_g <= nz_g / 2) ? kz_g : kz_g - nz_g;
            for (int ky = 0; ky < ny_g; ++ky) {
                const int fy = (ky <= ny_g / 2) ? ky : ky - ny_g;
                for (int kx = 0; kx < nx_c; ++kx) {
                    const int fx = kx;
                    const std::size_t idx = static_cast<std::size_t>(kx)
                        + nx_c * (static_cast<std::size_t>(ky)
                                  + static_cast<std::size_t>(ny_g) * k_loc);
                    const std::complex<Real>& uh = spec_buf[idx];
                    const Real weight = (fx == 0 || fx == nx_g / 2) ? 1.0 : 2.0;
                    const Real kmag = std::sqrt(static_cast<Real>(
                        fx * fx + fy * fy + fz * fz));
                    int b = static_cast<int>(std::round(kmag));
                    if (b > kmax) continue;
                    bin_sum_local[b] += 0.5 * weight *
                        (uh.real() * uh.real() + uh.imag() * uh.imag());
                }
            }
        }
    }

    std::vector<Real> bin_sum_global(kmax + 1, 0.0);
    MPI_Allreduce(bin_sum_local.data(), bin_sum_global.data(), kmax + 1,
                  MPI_DOUBLE, MPI_SUM, d.comm());

    ShellSpectrum sp;
    sp.k.resize(kmax + 1);
    sp.E.resize(kmax + 1);
    const Real total_cells = static_cast<Real>(nx_g) * ny_g * nz_g;
    const Real norm = 1.0 / (total_cells * total_cells);
    for (int b = 0; b <= kmax; ++b) {
        sp.k[b] = b * k_fund;
        sp.E[b] = bin_sum_global[b] * norm;
    }
    return sp;
}

HelmholtzResult helmholtz_decompose_mpi_dist(const State& U,
                                             const Grid& global_g,
                                             FFT3DPlanMPI& plan,
                                             const Domain& d) {
    const int nx_g = plan.nx_global();
    const int ny_g = plan.ny_global();
    const int nz_g = plan.nz_global();

    std::vector<CartDesc> cart;
    std::vector<SlabDesc> slab;
    gather_cart_descs(d, global_g, cart);
    gather_slab_descs(plan, d.size(), slab);

    const int my_kz0  = plan.local_z_start();
    const int my_nz_l = plan.local_nz();
    const int nx_c = nx_g / 2 + 1;

    // Each rank stores its slab's transformed û_x, û_y, û_z.
    const std::size_t slab_complex_words =
        static_cast<std::size_t>(my_nz_l) * ny_g * nx_c;
    std::vector<std::complex<Real>> uh(slab_complex_words);
    std::vector<std::complex<Real>> vh(slab_complex_words);
    std::vector<std::complex<Real>> wh(slab_complex_words);

    std::vector<double> local_buf;
    auto run_one = [&](int comp, std::vector<std::complex<Real>>& dst) {
        pack_velocity_component_local(U, comp, local_buf);
        redistribute_cart_to_slab(local_buf, cart, slab, plan, d);
        plan.forward();
        const std::complex<Real>* src = plan.complex_buf();
        std::copy(src, src + slab_complex_words, dst.begin());
    };
    run_one(0, uh);
    run_one(1, vh);
    run_one(2, wh);

    const Real two_pi_Lx = 2.0 * M_PI / global_g.lx;
    const Real two_pi_Ly = 2.0 * M_PI / global_g.ly;
    const Real two_pi_Lz = 2.0 * M_PI / global_g.lz;
    const Real k_fund    = std::min({two_pi_Lx, two_pi_Ly, two_pi_Lz});
    const int  kmax      = std::max({nx_g, ny_g, nz_g}) / 2;

    std::vector<Real> sol_sum_loc(kmax + 1, 0.0);
    std::vector<Real> dil_sum_loc(kmax + 1, 0.0);

    for (int k_loc = 0; k_loc < my_nz_l; ++k_loc) {
        const int kz_g = my_kz0 + k_loc;
        const int fz = (kz_g <= nz_g / 2) ? kz_g : kz_g - nz_g;
        for (int ky = 0; ky < ny_g; ++ky) {
            const int fy = (ky <= ny_g / 2) ? ky : ky - ny_g;
            for (int kx = 0; kx < nx_c; ++kx) {
                const int fx = kx;
                if (fx == 0 && fy == 0 && fz == 0) continue;
                const Real kxp = fx * two_pi_Lx;
                const Real kyp = fy * two_pi_Ly;
                const Real kzp = fz * two_pi_Lz;
                const Real kmag2 = kxp * kxp + kyp * kyp + kzp * kzp;
                if (kmag2 == 0) continue;
                const Real inv_kmag2 = 1.0 / kmag2;

                const std::size_t idx = static_cast<std::size_t>(kx)
                    + nx_c * (static_cast<std::size_t>(ky)
                              + static_cast<std::size_t>(ny_g) * k_loc);
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

                const Real weight = (fx == 0 || fx == nx_g / 2) ? 1.0 : 2.0;
                const Real e_sol = 0.5 * weight * (
                    sol_x.real() * sol_x.real() + sol_x.imag() * sol_x.imag()
                  + sol_y.real() * sol_y.real() + sol_y.imag() * sol_y.imag()
                  + sol_z.real() * sol_z.real() + sol_z.imag() * sol_z.imag());
                const Real e_dil = 0.5 * weight * (
                    dil_x.real() * dil_x.real() + dil_x.imag() * dil_x.imag()
                  + dil_y.real() * dil_y.real() + dil_y.imag() * dil_y.imag()
                  + dil_z.real() * dil_z.real() + dil_z.imag() * dil_z.imag());

                const Real kmag_int = std::sqrt(static_cast<Real>(
                    fx * fx + fy * fy + fz * fz));
                int b = static_cast<int>(std::round(kmag_int));
                if (b > kmax) continue;
                sol_sum_loc[b] += e_sol;
                dil_sum_loc[b] += e_dil;
            }
        }
    }

    std::vector<Real> sol_sum(kmax + 1, 0.0);
    std::vector<Real> dil_sum(kmax + 1, 0.0);
    MPI_Allreduce(sol_sum_loc.data(), sol_sum.data(), kmax + 1,
                  MPI_DOUBLE, MPI_SUM, d.comm());
    MPI_Allreduce(dil_sum_loc.data(), dil_sum.data(), kmax + 1,
                  MPI_DOUBLE, MPI_SUM, d.comm());

    HelmholtzResult r;
    const Real total_cells = static_cast<Real>(nx_g) * ny_g * nz_g;
    const Real norm = 1.0 / (total_cells * total_cells);
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

// ----------------------------------------------------------------------------
// Distributed cosine/sine spectrum for slip-wall domains.
//
// Mathematical setup. Slip-wall ghost cells are mirror-reflections with a
// sign-flip on the wall-normal momentum (bc/BC.cpp:40-93). Equivalently the
// field is the symmetric (resp. antisymmetric) periodic extension of period
// 2L across each wall. The natural spectral basis is therefore DCT-II for
// the symmetric (even-mirror) fields and DST-II for the antisymmetric
// (odd-mirror) fields. Wavenumbers k_n = n*pi/L for n = 0, 1, ..., N-1.
//
// We use FFTW REDFT10 (= DCT-II) and RODFT10 (= DST-II), unnormalized:
// forward * backward = 2N per axis -> 8N for the full 3D transform. The
// Parseval-correct energy factor on |coef|^2 is therefore 1/(2N_x*2N_y*2N_z)
// to match the periodic-FFT convention where E sums to <|u|^2/2>_volume.
// ----------------------------------------------------------------------------

namespace {

// R2R analog of redistribute_cart_to_slab. The r2r slab buffer has NO inner
// padding (row stride = nx_g), unlike the r2c case where it's 2*(nx_g/2+1).
void redistribute_cart_to_slab_r2r(const std::vector<double>& local_buf,
                                   const std::vector<CartDesc>& cart,
                                   const std::vector<SlabDesc>& slab,
                                   R2R3DPlanMPI& plan,
                                   const Domain& d) {
    const int size = d.size();
    const int my_rank = d.rank();
    const CartDesc& my = cart[my_rank];
    const SlabDesc& mys = slab[my_rank];
    const int nx_g = plan.nx_global();
    const int ny_g = plan.ny_global();
    const int row_stride   = nx_g;                   // r2r: no padding
    const std::size_t plane_stride =
        static_cast<std::size_t>(ny_g) * row_stride;

    std::vector<int> send_counts(size, 0), send_displs(size, 0);
    std::vector<int> recv_counts(size, 0), recv_displs(size, 0);
    long long total_send = 0, total_recv = 0;

    for (int r = 0; r < size; ++r) {
        const int sl = slab[r].k_start;
        const int sh = sl + slab[r].k_count;
        const int kk_lo = std::max(my.k_off, sl);
        const int kk_hi = std::min(my.k_off + my.nz, sh);
        const int kk_n  = std::max(0, kk_hi - kk_lo);
        send_counts[r] = my.nx * my.ny * kk_n;
        send_displs[r] = static_cast<int>(total_send);
        total_send += send_counts[r];
    }
    for (int s = 0; s < size; ++s) {
        const int kk_lo = std::max(cart[s].k_off, mys.k_start);
        const int kk_hi = std::min(cart[s].k_off + cart[s].nz,
                                   mys.k_start + mys.k_count);
        const int kk_n  = std::max(0, kk_hi - kk_lo);
        recv_counts[s] = cart[s].nx * cart[s].ny * kk_n;
        recv_displs[s] = static_cast<int>(total_recv);
        total_recv += recv_counts[s];
    }

    std::vector<double> send_buf(static_cast<std::size_t>(total_send));
    std::vector<double> recv_buf(static_cast<std::size_t>(total_recv));

    for (int r = 0; r < size; ++r) {
        if (send_counts[r] == 0) continue;
        const int sl = slab[r].k_start;
        const int sh = sl + slab[r].k_count;
        const int kk_lo = std::max(my.k_off, sl);
        const int kk_hi = std::min(my.k_off + my.nz, sh);
        double* dst = send_buf.data() + send_displs[r];
        std::size_t p = 0;
        for (int k_g = kk_lo; k_g < kk_hi; ++k_g) {
            const int k_loc = k_g - my.k_off;
            for (int j_loc = 0; j_loc < my.ny; ++j_loc) {
                const std::size_t row_base = static_cast<std::size_t>(my.nx)
                    * (static_cast<std::size_t>(j_loc)
                       + static_cast<std::size_t>(my.ny) * k_loc);
                for (int i_loc = 0; i_loc < my.nx; ++i_loc) {
                    dst[p++] = local_buf[row_base + i_loc];
                }
            }
        }
    }

    MPI_Alltoallv(send_buf.data(), send_counts.data(), send_displs.data(),
                  MPI_DOUBLE,
                  recv_buf.data(), recv_counts.data(), recv_displs.data(),
                  MPI_DOUBLE,
                  d.comm());

    double* rb = plan.real_buf();
    const std::size_t slab_real_words =
        static_cast<std::size_t>(plan.local_nz()) * plane_stride;
    std::fill(rb, rb + slab_real_words, 0.0);

    for (int s = 0; s < size; ++s) {
        if (recv_counts[s] == 0) continue;
        const int kk_lo = std::max(cart[s].k_off, mys.k_start);
        const int kk_hi = std::min(cart[s].k_off + cart[s].nz,
                                   mys.k_start + mys.k_count);
        const double* src = recv_buf.data() + recv_displs[s];
        std::size_t p = 0;
        for (int k_g = kk_lo; k_g < kk_hi; ++k_g) {
            const int k_loc = k_g - mys.k_start;
            for (int j_loc = 0; j_loc < cart[s].ny; ++j_loc) {
                const int j_g = cart[s].j_off + j_loc;
                const std::size_t row_base = static_cast<std::size_t>(row_stride) * j_g
                    + plane_stride * k_loc;
                for (int i_loc = 0; i_loc < cart[s].nx; ++i_loc) {
                    const int i_g = cart[s].i_off + i_loc;
                    rb[row_base + i_g] = src[p++];
                }
            }
        }
    }
}

void gather_slab_descs_r2r(const R2R3DPlanMPI& plan, int size,
                           std::vector<SlabDesc>& out) {
    int my[2] = {plan.local_z_start(), plan.local_nz()};
    std::vector<int> buf(static_cast<std::size_t>(size) * 2);
    MPI_Allgather(my, 2, MPI_INT, buf.data(), 2, MPI_INT, plan.comm());
    out.resize(static_cast<std::size_t>(size));
    for (int r = 0; r < size; ++r) {
        out[r].k_start = buf[2*r + 0];
        out[r].k_count = buf[2*r + 1];
    }
}

}  // namespace

ShellSpectrum velocity_spectrum_dct_mpi(const State& U, const Grid& global_g,
                                        R2R3DPlanMPI& plan_u,
                                        R2R3DPlanMPI& plan_v,
                                        R2R3DPlanMPI& plan_w,
                                        const Domain& d) {
    const int nx_g = plan_u.nx_global();
    const int ny_g = plan_u.ny_global();
    const int nz_g = plan_u.nz_global();

    // All three plans share the same domain, slab decomp, and comm
    // (only the per-axis kind differs).
    std::vector<CartDesc> cart;
    std::vector<SlabDesc> slab;
    gather_cart_descs(d, global_g, cart);
    gather_slab_descs_r2r(plan_u, d.size(), slab);

    // Cosine/sine fundamental wavenumber: pi/L (half of the periodic 2*pi/L).
    const Real k_fund_x = M_PI / global_g.lx;
    const Real k_fund_y = M_PI / global_g.ly;
    const Real k_fund_z = M_PI / global_g.lz;
    const Real k_fund   = std::min({k_fund_x, k_fund_y, k_fund_z});

    // Highest index along any axis is N - 1; bin index is round(|k|/k_fund),
    // so kmax_bin ~ sqrt(3) * (N-1) in axis units. Use the conservative bound.
    const int kmax = static_cast<int>(std::ceil(
        std::sqrt(3.0) * std::max({nx_g, ny_g, nz_g})));

    std::vector<Real> bin_sum_local(kmax + 1, 0.0);
    std::vector<double> local_buf;

    const int my_kz0  = plan_u.local_z_start();
    const int my_nz_l = plan_u.local_nz();

    R2R3DPlanMPI* plans[3] = {&plan_u, &plan_v, &plan_w};

    // FFTW r2r index-to-wavenumber offset (per axis):
    //   DCT-II (REDFT10): coefficient index k -> wavenumber k * pi/L
    //   DST-II (RODFT10): coefficient index k -> wavenumber (k+1) * pi/L
    // u uses DST on x, DCT on y/z;  v uses DST on y, DCT on x/z;
    // w uses DST on z, DCT on x/y. The +1 offset goes on whichever axis is DST.
    const int dst_off[3][3] = {
        {1, 0, 0},  // u: +1 on x
        {0, 1, 0},  // v: +1 on y
        {0, 0, 1},  // w: +1 on z
    };

    for (int comp = 0; comp < 3; ++comp) {
        pack_velocity_component_local(U, comp, local_buf);
        R2R3DPlanMPI& plan = *plans[comp];
        redistribute_cart_to_slab_r2r(local_buf, cart, slab, plan, d);
        plan.forward();

        const int ox = dst_off[comp][0];
        const int oy = dst_off[comp][1];
        const int oz = dst_off[comp][2];
        const double* spec_buf = plan.real_buf();
        // Real layout (no Hermitian fold, no padding):
        //   spec_buf[i + nx_g * (j + ny_g * k_local)]
        for (int k_loc = 0; k_loc < my_nz_l; ++k_loc) {
            const int nz_idx = my_kz0 + k_loc;
            const Real kz = (nz_idx + oz) * k_fund_z;
            for (int j = 0; j < ny_g; ++j) {
                const Real ky = (j + oy) * k_fund_y;
                for (int i = 0; i < nx_g; ++i) {
                    const Real kx = (i + ox) * k_fund_x;
                    const std::size_t idx = static_cast<std::size_t>(i)
                        + nx_g * (static_cast<std::size_t>(j)
                                  + static_cast<std::size_t>(ny_g) * k_loc);
                    const Real c = spec_buf[idx];
                    const Real kmag = std::sqrt(kx*kx + ky*ky + kz*kz);
                    int b = static_cast<int>(std::round(kmag / k_fund));
                    if (b > kmax) continue;
                    bin_sum_local[b] += 0.5 * c * c;
                }
            }
        }
    }

    std::vector<Real> bin_sum_global(kmax + 1, 0.0);
    MPI_Allreduce(bin_sum_local.data(), bin_sum_global.data(), kmax + 1,
                  MPI_DOUBLE, MPI_SUM, d.comm());

    // FFTW r2r normalization: REDFT10 / RODFT10 are unnormalized with
    // forward * backward = 2N per axis. The 3D transform therefore amplifies
    // |coef|^2 by (2 nx_g)*(2 ny_g)*(2 nz_g) = 8 N_total. Divide by that to
    // make Sum_bin E_bin = <(1/2) u_i u_i>_volume.
    const Real total_cells = static_cast<Real>(nx_g) * ny_g * nz_g;
    const Real norm = 1.0 / (8.0 * total_cells * total_cells);

    ShellSpectrum sp;
    sp.k.resize(kmax + 1);
    sp.E.resize(kmax + 1);
    for (int b = 0; b <= kmax; ++b) {
        sp.k[b] = b * k_fund;
        sp.E[b] = bin_sum_global[b] * norm;
    }
    return sp;
}

// ----------------------------------------------------------------------------
// Helmholtz decomposition in the slip-wall (DCT/DST mixed) basis.
//
// See header for the algorithm. Key implementation choices:
//   * The divergence array lives only on this rank's slab (size nx*ny*local_nz).
//   * Likewise phi_hat. No global gather is required.
//   * Inter-rank deps:
//       - For div_z at our slab's bottom (k_z = local_z_start) we need
//         ŵ at z = local_z_start - 1, which is rank R-1's TOP slice. So rank R
//         sends its own TOP slice up and receives R-1's TOP slice into a halo.
//       - For (u_dil_z) at our slab's top (k_z = local_z_start + local_nz - 1)
//         we need phi_hat at z = local_z_start + local_nz, which is rank R+1's
//         BOTTOM slice. Symmetric exchange.
//   * For the highest mode on each velocity component's DST axis (i = N-1 for u,
//     j = N-1 for v, k_z = N-1 globally for w), the corresponding phi_hat
//     coefficient at "DCT index N" is out of range -> u_dil component is zero
//     at that mode (the Nyquist-DST mode is fully solenoidal by construction).
// ----------------------------------------------------------------------------

HelmholtzResult helmholtz_decompose_dct_mpi(const State& U, const Grid& global_g,
                                            R2R3DPlanMPI& plan_u,
                                            R2R3DPlanMPI& plan_v,
                                            R2R3DPlanMPI& plan_w,
                                            const Domain& d) {
    const int nx_g = plan_u.nx_global();
    const int ny_g = plan_u.ny_global();
    const int nz_g = plan_u.nz_global();

    std::vector<CartDesc> cart;
    std::vector<SlabDesc> slab;
    gather_cart_descs(d, global_g, cart);
    gather_slab_descs_r2r(plan_u, d.size(), slab);

    const Real pi_Lx = M_PI / global_g.lx;
    const Real pi_Ly = M_PI / global_g.ly;
    const Real pi_Lz = M_PI / global_g.lz;
    const Real k_fund = std::min({pi_Lx, pi_Ly, pi_Lz});
    const int  kmax = static_cast<int>(std::ceil(
        std::sqrt(3.0) * std::max({nx_g, ny_g, nz_g})));

    // 1. Forward transforms. After this, plan_*.real_buf() holds the spectral
    // coefficients of u, v, w in their respective bases.
    std::vector<double> local_buf;
    auto fwd = [&](int comp, R2R3DPlanMPI& plan) {
        pack_velocity_component_local(U, comp, local_buf);
        redistribute_cart_to_slab_r2r(local_buf, cart, slab, plan, d);
        plan.forward();
    };
    fwd(0, plan_u);
    fwd(1, plan_v);
    fwd(2, plan_w);

    const int my_kz0  = plan_u.local_z_start();
    const int my_nz_l = plan_u.local_nz();
    const std::size_t plane = static_cast<std::size_t>(nx_g) * ny_g;
    const std::size_t slab_words = static_cast<std::size_t>(my_nz_l) * plane;

    const double* uhat = plan_u.real_buf();
    const double* vhat = plan_v.real_buf();
    const double* what = plan_w.real_buf();

    // 2a. Halo: receive ŵ's top z-slice from rank R-1 (so we can read ŵ at
    // local m_z = -1, i.e. global k_z = my_kz0 - 1). On rank 0 the halo is
    // unused because the m_z = 0 contribution to div is itself zero.
    std::vector<double> w_below(plane, 0.0);
    {
        const int rank = d.rank();
        const int size = d.size();
        std::vector<double> top_slice(plane);
        const std::size_t top_off = static_cast<std::size_t>(my_nz_l - 1) * plane;
        std::copy(what + top_off, what + top_off + plane, top_slice.begin());

        MPI_Request reqs[2];
        int nr = 0;
        if (rank + 1 < size) {
            MPI_Isend(top_slice.data(), static_cast<int>(plane), MPI_DOUBLE,
                      rank + 1, 5001, d.comm(), &reqs[nr++]);
        }
        if (rank > 0) {
            MPI_Irecv(w_below.data(), static_cast<int>(plane), MPI_DOUBLE,
                      rank - 1, 5001, d.comm(), &reqs[nr++]);
        }
        if (nr > 0) MPI_Waitall(nr, reqs, MPI_STATUSES_IGNORE);
    }

    // 3. Divergence in DCT_x × DCT_y × DCT_z, on this rank's slab.
    std::vector<double> div_hat(slab_words, 0.0);
#pragma omp parallel for collapse(2) schedule(static)
    for (int k_loc = 0; k_loc < my_nz_l; ++k_loc) {
        for (int j = 0; j < ny_g; ++j) {
            const int m_z = my_kz0 + k_loc;
            const Real kz_dct = m_z * pi_Lz;
            const Real ky_dct = j * pi_Ly;
            const std::size_t row_base = static_cast<std::size_t>(j) * nx_g
                + static_cast<std::size_t>(k_loc) * plane;
            for (int i = 0; i < nx_g; ++i) {
                const Real kx_dct = i * pi_Lx;
                Real acc = 0.0;
                // u contribution: û at DST index (m_x - 1)
                if (i > 0) {
                    const std::size_t idx_u = (i - 1)
                        + static_cast<std::size_t>(nx_g) * j
                        + static_cast<std::size_t>(plane) * k_loc;
                    acc += kx_dct * uhat[idx_u];
                }
                // v contribution: v̂ at DST index (m_y - 1)
                if (j > 0) {
                    const std::size_t idx_v = i
                        + static_cast<std::size_t>(nx_g) * (j - 1)
                        + static_cast<std::size_t>(plane) * k_loc;
                    acc += ky_dct * vhat[idx_v];
                }
                // w contribution: ŵ at DST index (m_z - 1). m_z = 0 -> no contribution.
                if (m_z > 0) {
                    if (k_loc > 0) {
                        const std::size_t idx_w = i
                            + static_cast<std::size_t>(nx_g) * j
                            + static_cast<std::size_t>(plane) * (k_loc - 1);
                        acc += kz_dct * what[idx_w];
                    } else {
                        // m_z == my_kz0 > 0; read from rank R-1's halo.
                        acc += kz_dct *
                            w_below[i + static_cast<std::size_t>(nx_g) * j];
                    }
                }
                div_hat[row_base + i] = acc;
            }
        }
    }

    // 4. Solve Poisson in DCT space: phi_hat = -div_hat / |k_DCT|^2.
    std::vector<double> phi_hat(slab_words, 0.0);
#pragma omp parallel for collapse(2) schedule(static)
    for (int k_loc = 0; k_loc < my_nz_l; ++k_loc) {
        for (int j = 0; j < ny_g; ++j) {
            const int m_z = my_kz0 + k_loc;
            const Real kz = m_z * pi_Lz;
            const Real ky = j * pi_Ly;
            const std::size_t row_base = static_cast<std::size_t>(j) * nx_g
                + static_cast<std::size_t>(k_loc) * plane;
            for (int i = 0; i < nx_g; ++i) {
                const Real kx = i * pi_Lx;
                const Real k2 = kx * kx + ky * ky + kz * kz;
                if (k2 == 0.0) continue;   // (0,0,0): phi undefined
                phi_hat[row_base + i] = -div_hat[row_base + i] / k2;
            }
        }
    }

    // 5a. Halo: receive phi_hat's BOTTOM z-slice from rank R+1 (so we can
    // read phi at local m_z = my_nz_l, i.e. global k_z = my_kz0 + my_nz_l).
    // On the top rank the halo stays at zero (no rank R+1; the highest
    // global m_z = nz_g - 1 cannot reach phi at m_z = nz_g anyway).
    std::vector<double> phi_above(plane, 0.0);
    {
        const int rank = d.rank();
        const int size = d.size();
        std::vector<double> bot_slice(plane);
        std::copy(phi_hat.begin(), phi_hat.begin() + plane, bot_slice.begin());
        MPI_Request reqs[2];
        int nr = 0;
        if (rank > 0) {
            MPI_Isend(bot_slice.data(), static_cast<int>(plane), MPI_DOUBLE,
                      rank - 1, 5002, d.comm(), &reqs[nr++]);
        }
        if (rank + 1 < size) {
            MPI_Irecv(phi_above.data(), static_cast<int>(plane), MPI_DOUBLE,
                      rank + 1, 5002, d.comm(), &reqs[nr++]);
        }
        if (nr > 0) MPI_Waitall(nr, reqs, MPI_STATUSES_IGNORE);
    }

    // 6. Recover u_dil_x, u_dil_y, u_dil_z; form u_sol; bin per-component
    // energies into shells. Each velocity component's wavenumbers are shifted
    // by +1 on its own DST axis. The u_dil contribution to (component c) at
    // its grid index (i,j,k) is -k_axis * phi_hat at the DCT index (i+1 for u,
    // j+1 for v, (k_loc+1) for w).
    std::vector<Real> sol_sum_loc(kmax + 1, 0.0);
    std::vector<Real> dil_sum_loc(kmax + 1, 0.0);

    // ----- u-component (DST_x × DCT_y × DCT_z) -----
    for (int k_loc = 0; k_loc < my_nz_l; ++k_loc) {
        const Real kz = (my_kz0 + k_loc) * pi_Lz;
        for (int j = 0; j < ny_g; ++j) {
            const Real ky = j * pi_Ly;
            for (int i = 0; i < nx_g; ++i) {
                const Real kx = (i + 1) * pi_Lx;
                const std::size_t idx = i
                    + static_cast<std::size_t>(nx_g) * j
                    + plane * k_loc;
                const Real uhv = uhat[idx];
                Real u_dil = 0.0;
                if (i + 1 < nx_g) {
                    const std::size_t pidx = (i + 1)
                        + static_cast<std::size_t>(nx_g) * j
                        + plane * k_loc;
                    u_dil = -kx * phi_hat[pidx];
                }
                const Real u_sol = uhv - u_dil;
                const Real kmag = std::sqrt(kx*kx + ky*ky + kz*kz);
                int b = static_cast<int>(std::round(kmag / k_fund));
                if (b > kmax) continue;
                sol_sum_loc[b] += 0.5 * u_sol * u_sol;
                dil_sum_loc[b] += 0.5 * u_dil * u_dil;
            }
        }
    }

    // ----- v-component (DCT_x × DST_y × DCT_z) -----
    for (int k_loc = 0; k_loc < my_nz_l; ++k_loc) {
        const Real kz = (my_kz0 + k_loc) * pi_Lz;
        for (int j = 0; j < ny_g; ++j) {
            const Real ky = (j + 1) * pi_Ly;
            for (int i = 0; i < nx_g; ++i) {
                const Real kx = i * pi_Lx;
                const std::size_t idx = i
                    + static_cast<std::size_t>(nx_g) * j
                    + plane * k_loc;
                const Real vhv = vhat[idx];
                Real v_dil = 0.0;
                if (j + 1 < ny_g) {
                    const std::size_t pidx = i
                        + static_cast<std::size_t>(nx_g) * (j + 1)
                        + plane * k_loc;
                    v_dil = -ky * phi_hat[pidx];
                }
                const Real v_sol = vhv - v_dil;
                const Real kmag = std::sqrt(kx*kx + ky*ky + kz*kz);
                int b = static_cast<int>(std::round(kmag / k_fund));
                if (b > kmax) continue;
                sol_sum_loc[b] += 0.5 * v_sol * v_sol;
                dil_sum_loc[b] += 0.5 * v_dil * v_dil;
            }
        }
    }

    // ----- w-component (DCT_x × DCT_y × DST_z) -----
    for (int k_loc = 0; k_loc < my_nz_l; ++k_loc) {
        const int k_z_idx_dst = my_kz0 + k_loc;          // DST index along z
        const Real kz = (k_z_idx_dst + 1) * pi_Lz;       // wavenumber for w
        // phi at DCT index (k_z_idx_dst + 1):
        //   if k_z_idx_dst + 1 is within our slab, read phi_hat locally
        //   if k_z_idx_dst + 1 == my_kz0 + my_nz_l (one past top), use phi_above halo
        //   if k_z_idx_dst + 1 > nz_g - 1 (top mode), set u_dil = 0
        const bool phi_local = (k_loc + 1 < my_nz_l);
        const bool phi_halo  = (k_loc + 1 == my_nz_l && k_z_idx_dst + 1 < nz_g);
        for (int j = 0; j < ny_g; ++j) {
            const Real ky = j * pi_Ly;
            for (int i = 0; i < nx_g; ++i) {
                const Real kx = i * pi_Lx;
                const std::size_t idx = i
                    + static_cast<std::size_t>(nx_g) * j
                    + plane * k_loc;
                const Real whv = what[idx];
                Real w_dil = 0.0;
                if (phi_local) {
                    const std::size_t pidx = i
                        + static_cast<std::size_t>(nx_g) * j
                        + plane * (k_loc + 1);
                    w_dil = -kz * phi_hat[pidx];
                } else if (phi_halo) {
                    w_dil = -kz * phi_above[i + static_cast<std::size_t>(nx_g) * j];
                }
                const Real w_sol = whv - w_dil;
                const Real kmag = std::sqrt(kx*kx + ky*ky + kz*kz);
                int b = static_cast<int>(std::round(kmag / k_fund));
                if (b > kmax) continue;
                sol_sum_loc[b] += 0.5 * w_sol * w_sol;
                dil_sum_loc[b] += 0.5 * w_dil * w_dil;
            }
        }
    }

    // 7. Reduce + normalize.
    std::vector<Real> sol_sum(kmax + 1, 0.0);
    std::vector<Real> dil_sum(kmax + 1, 0.0);
    MPI_Allreduce(sol_sum_loc.data(), sol_sum.data(), kmax + 1,
                  MPI_DOUBLE, MPI_SUM, d.comm());
    MPI_Allreduce(dil_sum_loc.data(), dil_sum.data(), kmax + 1,
                  MPI_DOUBLE, MPI_SUM, d.comm());

    const Real total_cells = static_cast<Real>(nx_g) * ny_g * nz_g;
    const Real norm = 1.0 / (8.0 * total_cells * total_cells);

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

#endif

}  // namespace blast
