#include "numerics/HyperdissipationSpectral.hpp"

#include "core/Grid.hpp"
#include "core/State.hpp"
#include "diagnostics/FFT.hpp"

#ifdef BLAST_MPI
#include "parallel/Domain.hpp"

#include <fftw3.h>
#include <fftw3-mpi.h>
#include <omp.h>
#endif

#include <cmath>
#include <cstddef>

namespace blast {

namespace {

// An axis carries the wall-normal momentum for ConsVar v iff v == RHOU + axis
// (axis 0->RHOU, 1->RHOV, 2->RHOW). On such an axis the variable is odd at
// the wall and uses DST-II; everything else uses DCT-II.
constexpr r2r::Kind kind_for(int v, int axis) {
    return (v == RHOU + axis) ? r2r::Kind::DST_II : r2r::Kind::DCT_II;
}

constexpr r2r::Kind inverse_kind(r2r::Kind k) {
    return (k == r2r::Kind::DCT_II) ? r2r::Kind::DCT_III : r2r::Kind::DST_III;
}

}  // namespace

HyperdissipationSpectral::HyperdissipationSpectral(int nx, int ny, int nz,
                                                   SpectralBCMode mode)
    : nx_(nx), ny_(ny), nz_(nz), mode_(mode),
      real_buf_(static_cast<std::size_t>(nx) * ny * nz) {
    if (mode_ == SpectralBCMode::Periodic) {
        dft_fwd_ = std::make_unique<FFT3DPlan>(nx, ny, nz);
        dft_inv_ = std::make_unique<FFT3DInversePlan>(nx, ny, nz);
        spec_buf_.resize(static_cast<std::size_t>(nx / 2 + 1) * ny * nz);
    } else {
        for (int v = 0; v < NCONS; ++v) {
            const auto kx = kind_for(v, 0);
            const auto ky = kind_for(v, 1);
            const auto kz = kind_for(v, 2);
            r2r_fwd_[v] = std::make_unique<R2R3DPlan>(nx, ny, nz, kz, ky, kx);
            r2r_inv_[v] = std::make_unique<R2R3DPlan>(
                nx, ny, nz, inverse_kind(kz), inverse_kind(ky), inverse_kind(kx));
        }
    }
}

HyperdissipationSpectral::~HyperdissipationSpectral() = default;

void HyperdissipationSpectral::apply(const State& U, const Grid& g,
                                     Real nu4, Real nu6, State& Rhs) {
    if (nu4 <= 0.0 && nu6 <= 0.0) return;
    if (mode_ == SpectralBCMode::Periodic) apply_periodic_(U, g, nu4, nu6, Rhs);
    else                                   apply_slip_wall_(U, g, nu4, nu6, Rhs);
}

void HyperdissipationSpectral::apply_periodic_(const State& U, const Grid& g,
                                               Real nu4, Real nu6, State& Rhs) {
    const int nx = nx_, ny = ny_, nz = nz_;
    const Real inv_N = 1.0 / (static_cast<Real>(nx) * ny * nz);
    const Real kfx = 2.0 * M_PI / g.lx;
    const Real kfy = 2.0 * M_PI / g.ly;
    const Real kfz = 2.0 * M_PI / g.lz;
    const int nx_c = nx / 2 + 1;

    auto signed_freq = [](int idx, int N) {
        return (idx <= N / 2) ? idx : idx - N;
    };

    for (int v = 0; v < NCONS; ++v) {
        const Field3D& Uv = U[v];

#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
#pragma omp simd
                for (int i = 0; i < nx; ++i) {
                    const std::size_t idx =
                        static_cast<std::size_t>(i)
                        + nx * (static_cast<std::size_t>(j) + ny * k);
                    real_buf_[idx] = Uv(i, j, k);
                }

        dft_fwd_->forward(real_buf_.data(), spec_buf_.data());

#pragma omp parallel for collapse(2) schedule(static)
        for (int kz = 0; kz < nz; ++kz)
            for (int ky = 0; ky < ny; ++ky) {
                const Real fz = kfz * signed_freq(kz, nz);
                const Real fy = kfy * signed_freq(ky, ny);
                for (int kx = 0; kx < nx_c; ++kx) {
                    const Real fx = kfx * kx;
                    const Real k2 = fx * fx + fy * fy + fz * fz;
                    const Real k4 = k2 * k2;
                    const Real mult = -(nu4 * k4 + nu6 * k4 * k2) * inv_N;
                    const std::size_t idx =
                        static_cast<std::size_t>(kx)
                        + nx_c * (static_cast<std::size_t>(ky) + ny * kz);
                    spec_buf_[idx] *= mult;
                }
            }

        dft_inv_->inverse(spec_buf_.data(), real_buf_.data());

        Field3D& Rv = Rhs[v];
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
#pragma omp simd
                for (int i = 0; i < nx; ++i) {
                    const std::size_t idx =
                        static_cast<std::size_t>(i)
                        + nx * (static_cast<std::size_t>(j) + ny * k);
                    Rv(i, j, k) += real_buf_[idx];
                }
    }
}

void HyperdissipationSpectral::apply_slip_wall_(const State& U, const Grid& g,
                                                Real nu4, Real nu6, State& Rhs) {
    const int nx = nx_, ny = ny_, nz = nz_;
    // R2R round-trip normalization: 2 N per axis -> 8 N_x N_y N_z total.
    const Real inv_2N3 = 1.0 / (8.0 * static_cast<Real>(nx) * ny * nz);

    // Per-axis k_n tables for both kinds.  DCT-II coefficient index n
    // corresponds to basis cos(pi n (i+1/2)/N), continuous wavenumber
    // k_n = pi n / L.  DST-II index n corresponds to sin(pi (n+1)(i+1/2)/N),
    // k_n = pi (n+1) / L.
    const Real cx = M_PI / g.lx;
    const Real cy = M_PI / g.ly;
    const Real cz = M_PI / g.lz;
    std::vector<Real> k2x_dct(nx), k2x_dst(nx);
    std::vector<Real> k2y_dct(ny), k2y_dst(ny);
    std::vector<Real> k2z_dct(nz), k2z_dst(nz);
    for (int n = 0; n < nx; ++n) {
        const Real kd = cx * n;       const Real ks = cx * (n + 1);
        k2x_dct[n] = kd * kd;         k2x_dst[n] = ks * ks;
    }
    for (int n = 0; n < ny; ++n) {
        const Real kd = cy * n;       const Real ks = cy * (n + 1);
        k2y_dct[n] = kd * kd;         k2y_dst[n] = ks * ks;
    }
    for (int n = 0; n < nz; ++n) {
        const Real kd = cz * n;       const Real ks = cz * (n + 1);
        k2z_dct[n] = kd * kd;         k2z_dst[n] = ks * ks;
    }

    for (int v = 0; v < NCONS; ++v) {
        const Field3D& Uv = U[v];

#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
#pragma omp simd
                for (int i = 0; i < nx; ++i) {
                    const std::size_t idx =
                        static_cast<std::size_t>(i)
                        + nx * (static_cast<std::size_t>(j) + ny * k);
                    real_buf_[idx] = Uv(i, j, k);
                }

        r2r_fwd_[v]->execute(real_buf_.data());

        // Pick the per-axis k^2 table for this variable's kind.
        const Real* k2x = (kind_for(v, 0) == r2r::Kind::DST_II)
                              ? k2x_dst.data() : k2x_dct.data();
        const Real* k2y = (kind_for(v, 1) == r2r::Kind::DST_II)
                              ? k2y_dst.data() : k2y_dct.data();
        const Real* k2z = (kind_for(v, 2) == r2r::Kind::DST_II)
                              ? k2z_dst.data() : k2z_dct.data();

#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j) {
                const Real k2_jk = k2z[k] + k2y[j];
#pragma omp simd
                for (int i = 0; i < nx; ++i) {
                    const Real k2 = k2_jk + k2x[i];
                    const Real k4 = k2 * k2;
                    const Real mult = -(nu4 * k4 + nu6 * k4 * k2) * inv_2N3;
                    const std::size_t idx =
                        static_cast<std::size_t>(i)
                        + nx * (static_cast<std::size_t>(j) + ny * k);
                    real_buf_[idx] *= mult;
                }
            }

        r2r_inv_[v]->execute(real_buf_.data());

        Field3D& Rv = Rhs[v];
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
#pragma omp simd
                for (int i = 0; i < nx; ++i) {
                    const std::size_t idx =
                        static_cast<std::size_t>(i)
                        + nx * (static_cast<std::size_t>(j) + ny * k);
                    Rv(i, j, k) += real_buf_[idx];
                }
    }
}

#ifdef BLAST_MPI

namespace {

constexpr int fftw_kind_for(int v, int axis) {
    // Mirrors kind_for() in the serial slip-wall code: wall-normal momentum
    // is DST-II on its axis; everything else is DCT-II.
    return (v == RHOU + axis) ? FFTW_RODFT10 : FFTW_REDFT10;
}

constexpr int fftw_inverse_kind(int fwd) {
    return (fwd == FFTW_REDFT10) ? FFTW_REDFT01 : FFTW_RODFT01;
}

}  // namespace

HyperdissipationSpectralMpi::HyperdissipationSpectralMpi(
    const Grid& global_grid, const Domain& d, SpectralBCMode mode)
    : d_(&d), mode_(mode),
      nx_g_(static_cast<int>(global_grid.nx)),
      ny_g_(static_cast<int>(global_grid.ny)),
      nz_g_(static_cast<int>(global_grid.nz)),
      lx_(global_grid.lx), ly_(global_grid.ly), lz_(global_grid.lz) {
    Grid lg = d.local_grid(global_grid);
    nx_l_ = lg.nx; ny_l_ = lg.ny; nz_l_ = lg.nz;
    cart_buf_.assign(
        static_cast<std::size_t>(nx_l_) * ny_l_ * nz_l_, 0.0);

    init_fftw_threads();
    init_fftw_mpi();
    fftw_plan_with_nthreads(omp_get_max_threads());

    if (mode_ == SpectralBCMode::Periodic) {
        ptrdiff_t local_n0, local_0_start;
        const ptrdiff_t alloc = fftw_mpi_local_size_3d(
            nz_g_, ny_g_, nx_g_ / 2 + 1, d.comm(),
            &local_n0, &local_0_start);
        local_nz_      = static_cast<long long>(local_n0);
        local_z_start_ = static_cast<long long>(local_0_start);
        row_stride_    = 2 * (nx_g_ / 2 + 1);

        slab_buf_ = fftw_alloc_real(2 * static_cast<std::size_t>(alloc));

        // Both plans operate IN-PLACE on slab_buf_ so we can chain them
        // without a copy (and so the new-array execute rules permit reuse).
        fwd_plan_ = static_cast<void*>(fftw_mpi_plan_dft_r2c_3d(
            nz_g_, ny_g_, nx_g_,
            slab_buf_,
            reinterpret_cast<fftw_complex*>(slab_buf_),
            d.comm(), FFTW_ESTIMATE));
        inv_plan_ = static_cast<void*>(fftw_mpi_plan_dft_c2r_3d(
            nz_g_, ny_g_, nx_g_,
            reinterpret_cast<fftw_complex*>(slab_buf_),
            slab_buf_,
            d.comm(), FFTW_ESTIMATE));
    } else {
        // SlipWall: R2R, full nx_g inner dim (no Hermitian fold, no padding).
        ptrdiff_t local_n0, local_0_start;
        const ptrdiff_t alloc = fftw_mpi_local_size_3d(
            nz_g_, ny_g_, nx_g_, d.comm(),
            &local_n0, &local_0_start);
        local_nz_      = static_cast<long long>(local_n0);
        local_z_start_ = static_cast<long long>(local_0_start);
        row_stride_    = nx_g_;

        slab_buf_ = fftw_alloc_real(static_cast<std::size_t>(alloc));

        // Per-variable forward + inverse R2R plans, all in-place on
        // slab_buf_. Each plan has a kind triple determined by the
        // variable's reflection parity on each axis.
        for (int v = 0; v < NCONS; ++v) {
            const fftw_r2r_kind kf[3] = {
                static_cast<fftw_r2r_kind>(fftw_kind_for(v, 2)),  // z
                static_cast<fftw_r2r_kind>(fftw_kind_for(v, 1)),  // y
                static_cast<fftw_r2r_kind>(fftw_kind_for(v, 0)),  // x
            };
            const fftw_r2r_kind ki[3] = {
                static_cast<fftw_r2r_kind>(fftw_inverse_kind(kf[0])),
                static_cast<fftw_r2r_kind>(fftw_inverse_kind(kf[1])),
                static_cast<fftw_r2r_kind>(fftw_inverse_kind(kf[2])),
            };
            r2r_fwd_plans_[v] = static_cast<void*>(fftw_mpi_plan_r2r_3d(
                nz_g_, ny_g_, nx_g_, slab_buf_, slab_buf_,
                d.comm(), kf[0], kf[1], kf[2], FFTW_ESTIMATE));
            r2r_inv_plans_[v] = static_cast<void*>(fftw_mpi_plan_r2r_3d(
                nz_g_, ny_g_, nx_g_, slab_buf_, slab_buf_,
                d.comm(), ki[0], ki[1], ki[2], FFTW_ESTIMATE));
        }

        // Per-axis k^2 tables. DCT-II index n -> pi n / L; DST-II index n
        // -> pi (n+1) / L. Eigenvalue of nabla^2 is -k^2; we store +k^2.
        const Real cx = M_PI / lx_;
        const Real cy = M_PI / ly_;
        const Real cz = M_PI / lz_;
        k2x_dct_.resize(nx_g_); k2x_dst_.resize(nx_g_);
        k2y_dct_.resize(ny_g_); k2y_dst_.resize(ny_g_);
        k2z_dct_.resize(nz_g_); k2z_dst_.resize(nz_g_);
        for (int n = 0; n < nx_g_; ++n) {
            const Real kd = cx * n, ks = cx * (n + 1);
            k2x_dct_[n] = kd * kd; k2x_dst_[n] = ks * ks;
        }
        for (int n = 0; n < ny_g_; ++n) {
            const Real kd = cy * n, ks = cy * (n + 1);
            k2y_dct_[n] = kd * kd; k2y_dst_[n] = ks * ks;
        }
        for (int n = 0; n < nz_g_; ++n) {
            const Real kd = cz * n, ks = cz * (n + 1);
            k2z_dct_[n] = kd * kd; k2z_dst_[n] = ks * ks;
        }
    }

    gather_cart_descs(d, global_grid, cart_descs_);
    gather_slab_descs(d.comm(),
                      static_cast<int>(local_z_start_),
                      static_cast<int>(local_nz_),
                      slab_descs_);
}

HyperdissipationSpectralMpi::~HyperdissipationSpectralMpi() {
    if (fwd_plan_) fftw_destroy_plan(static_cast<fftw_plan>(fwd_plan_));
    if (inv_plan_) fftw_destroy_plan(static_cast<fftw_plan>(inv_plan_));
    for (int v = 0; v < NCONS; ++v) {
        if (r2r_fwd_plans_[v])
            fftw_destroy_plan(static_cast<fftw_plan>(r2r_fwd_plans_[v]));
        if (r2r_inv_plans_[v])
            fftw_destroy_plan(static_cast<fftw_plan>(r2r_inv_plans_[v]));
    }
    if (slab_buf_) fftw_free(slab_buf_);
}

void HyperdissipationSpectralMpi::apply(const State& U, const Grid& g,
                                        Real nu4, Real nu6, State& Rhs) {
    if (nu4 <= 0.0 && nu6 <= 0.0) return;
    (void)g;
    if (mode_ == SpectralBCMode::Periodic) apply_periodic_(U, nu4, nu6, Rhs);
    else                                   apply_slip_wall_(U, nu4, nu6, Rhs);
}

void HyperdissipationSpectralMpi::apply_periodic_(const State& U,
                                                  Real nu4, Real nu6,
                                                  State& Rhs) {
    const int nx_l = nx_l_, ny_l = ny_l_, nz_l = nz_l_;
    const int nx_g = nx_g_, ny_g = ny_g_, nz_g = nz_g_;
    const Real inv_N = 1.0 / (static_cast<Real>(nx_g) * ny_g * nz_g);
    const Real kfx = 2.0 * M_PI / lx_;
    const Real kfy = 2.0 * M_PI / ly_;
    const Real kfz = 2.0 * M_PI / lz_;
    const int nx_c = nx_g / 2 + 1;
    const int my_kz0  = static_cast<int>(local_z_start_);
    const int my_nz_l = static_cast<int>(local_nz_);

    auto signed_freq = [](int idx, int N) {
        return (idx <= N / 2) ? idx : idx - N;
    };

    for (int v = 0; v < NCONS; ++v) {
        const Field3D& Uv = U[v];

        // 1. Pack local interior, i-fastest.
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 0; k < nz_l; ++k)
            for (int j = 0; j < ny_l; ++j)
#pragma omp simd
                for (int i = 0; i < nx_l; ++i) {
                    const std::size_t idx =
                        static_cast<std::size_t>(i)
                        + nx_l * (static_cast<std::size_t>(j) + ny_l * k);
                    cart_buf_[idx] = Uv(i, j, k);
                }

        // 2. Cart -> slab redistribute.
        redistribute_cart_to_slab(
            cart_buf_.data(), slab_buf_,
            nx_g, ny_g, row_stride_, my_nz_l, my_kz0,
            cart_descs_, slab_descs_, *d_);

        // 3. Forward r2c (in-place).
        fftw_mpi_execute_dft_r2c(
            static_cast<fftw_plan>(fwd_plan_),
            slab_buf_,
            reinterpret_cast<fftw_complex*>(slab_buf_));

        // 4. Spectral multiplier on this rank's local slab modes.
        auto* spec = reinterpret_cast<std::complex<Real>*>(slab_buf_);
#pragma omp parallel for collapse(2) schedule(static)
        for (int k_loc = 0; k_loc < my_nz_l; ++k_loc)
            for (int ky = 0; ky < ny_g; ++ky) {
                const int kz_g = my_kz0 + k_loc;
                const Real fz = kfz * signed_freq(kz_g, nz_g);
                const Real fy = kfy * signed_freq(ky, ny_g);
                for (int kx = 0; kx < nx_c; ++kx) {
                    const Real fx = kfx * kx;
                    const Real k2 = fx * fx + fy * fy + fz * fz;
                    const Real k4 = k2 * k2;
                    const Real mult =
                        -(nu4 * k4 + nu6 * k4 * k2) * inv_N;
                    const std::size_t idx =
                        static_cast<std::size_t>(kx)
                        + nx_c * (static_cast<std::size_t>(ky)
                                  + static_cast<std::size_t>(ny_g) * k_loc);
                    spec[idx] *= mult;
                }
            }

        // 5. Inverse c2r (in-place on the same buffer).
        fftw_mpi_execute_dft_c2r(
            static_cast<fftw_plan>(inv_plan_),
            reinterpret_cast<fftw_complex*>(slab_buf_),
            slab_buf_);

        // 6. Slab -> Cart redistribute.
        redistribute_slab_to_cart(
            slab_buf_, cart_buf_.data(),
            nx_g, ny_g, row_stride_, my_nz_l, my_kz0,
            cart_descs_, slab_descs_, *d_);

        // 7. Accumulate into Rhs.
        Field3D& Rv = Rhs[v];
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 0; k < nz_l; ++k)
            for (int j = 0; j < ny_l; ++j)
#pragma omp simd
                for (int i = 0; i < nx_l; ++i) {
                    const std::size_t idx =
                        static_cast<std::size_t>(i)
                        + nx_l * (static_cast<std::size_t>(j) + ny_l * k);
                    Rv(i, j, k) += cart_buf_[idx];
                }
    }
}

void HyperdissipationSpectralMpi::apply_slip_wall_(const State& U,
                                                   Real nu4, Real nu6,
                                                   State& Rhs) {
    const int nx_l = nx_l_, ny_l = ny_l_, nz_l = nz_l_;
    const int nx_g = nx_g_, ny_g = ny_g_;
    // R2R round-trip normalization: 2N per axis -> 8 N_x N_y N_z total.
    const Real inv_2N3 = 1.0 / (8.0 * static_cast<Real>(nx_g) * ny_g_ * nz_g_);
    const int my_kz0  = static_cast<int>(local_z_start_);
    const int my_nz_l = static_cast<int>(local_nz_);

    for (int v = 0; v < NCONS; ++v) {
        const Field3D& Uv = U[v];

        // 1. Pack local interior, i-fastest.
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 0; k < nz_l; ++k)
            for (int j = 0; j < ny_l; ++j)
#pragma omp simd
                for (int i = 0; i < nx_l; ++i) {
                    const std::size_t idx =
                        static_cast<std::size_t>(i)
                        + nx_l * (static_cast<std::size_t>(j) + ny_l * k);
                    cart_buf_[idx] = Uv(i, j, k);
                }

        // 2. Cart -> slab redistribute. R2R has no inner-dim padding, so
        // row_stride_ == nx_g and the same redistribute helper works.
        redistribute_cart_to_slab(
            cart_buf_.data(), slab_buf_,
            nx_g, ny_g, row_stride_, my_nz_l, my_kz0,
            cart_descs_, slab_descs_, *d_);

        // 3. Forward R2R for this variable (in-place).
        fftw_mpi_execute_r2r(
            static_cast<fftw_plan>(r2r_fwd_plans_[v]),
            slab_buf_, slab_buf_);

        // 4. Spectral multiplier on this rank's local slab modes.
        // Pick per-axis k^2 table by the variable's kind on that axis.
        const Real* k2x = (v == RHOU + 0) ? k2x_dst_.data() : k2x_dct_.data();
        const Real* k2y = (v == RHOU + 1) ? k2y_dst_.data() : k2y_dct_.data();
        const Real* k2z = (v == RHOU + 2) ? k2z_dst_.data() : k2z_dct_.data();

#pragma omp parallel for collapse(2) schedule(static)
        for (int k_loc = 0; k_loc < my_nz_l; ++k_loc)
            for (int j = 0; j < ny_g; ++j) {
                const Real k2_jk = k2z[my_kz0 + k_loc] + k2y[j];
#pragma omp simd
                for (int i = 0; i < nx_g; ++i) {
                    const Real k2 = k2_jk + k2x[i];
                    const Real k4 = k2 * k2;
                    const Real mult =
                        -(nu4 * k4 + nu6 * k4 * k2) * inv_2N3;
                    const std::size_t idx =
                        static_cast<std::size_t>(i)
                        + nx_g * (static_cast<std::size_t>(j) + ny_g * k_loc);
                    slab_buf_[idx] *= mult;
                }
            }

        // 5. Inverse R2R (in-place on the same buffer).
        fftw_mpi_execute_r2r(
            static_cast<fftw_plan>(r2r_inv_plans_[v]),
            slab_buf_, slab_buf_);

        // 6. Slab -> Cart redistribute.
        redistribute_slab_to_cart(
            slab_buf_, cart_buf_.data(),
            nx_g, ny_g, row_stride_, my_nz_l, my_kz0,
            cart_descs_, slab_descs_, *d_);

        // 7. Accumulate into Rhs.
        Field3D& Rv = Rhs[v];
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 0; k < nz_l; ++k)
            for (int j = 0; j < ny_l; ++j)
#pragma omp simd
                for (int i = 0; i < nx_l; ++i) {
                    const std::size_t idx =
                        static_cast<std::size_t>(i)
                        + nx_l * (static_cast<std::size_t>(j) + ny_l * k);
                    Rv(i, j, k) += cart_buf_[idx];
                }
    }
}

#endif  // BLAST_MPI

}  // namespace blast
