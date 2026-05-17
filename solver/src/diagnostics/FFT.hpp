#pragma once

#include "core/Types.hpp"

#ifdef BLAST_MPI
#include <mpi.h>
#endif

#include <complex>
#include <cstddef>
#include <vector>

namespace blast {

// RAII wrapper around an FFTW3 real-to-complex 3D plan with OpenMP threading.
// Out-of-place transform: input is a contiguous nx*ny*nz real buffer (i-fastest);
// output is a contiguous nx*ny*(nz/2+1) std::complex<Real> buffer.
class FFT3DPlan {
public:
    FFT3DPlan(int nx, int ny, int nz);
    ~FFT3DPlan();
    FFT3DPlan(const FFT3DPlan&) = delete;
    FFT3DPlan& operator=(const FFT3DPlan&) = delete;

    void forward(const Real* in, std::complex<Real>* out) const;

    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int nz() const { return nz_; }
    std::size_t complex_size() const {
        return static_cast<std::size_t>(nx_) * ny_ * (nz_ / 2 + 1);
    }

private:
    int   nx_, ny_, nz_;
    void* plan_ = nullptr;
};

// One-shot initialization of FFTW thread support; idempotent.
void init_fftw_threads();

#ifdef BLAST_MPI
// Distributed FFTW3-MPI r2c plan. Uses 1D slab decomposition along the
// OUTERMOST (z) dimension: each rank owns local_nz() consecutive z-planes
// starting at local_z_start(). In-place transform: one allocated buffer of
// size 2 * alloc_local() doubles, reinterpreted as complex on output.
//
// Real-space layout (in-place padded, k-fastest-after-padding):
//   real_buf[i + (2*(nx/2+1)) * (j + ny * k_local)]
//     for i in [0, nx), j in [0, ny), k_local in [0, local_nz()).
// Complex-space layout (post-forward):
//   complex_buf[kx + (nx/2+1) * (ky + ny * kz_local)]
//     for kx in [0, nx/2], ky in [0, ny), kz_local in [0, local_nz()).
//
// The communicator is reused as-is (typically the Cartesian comm from
// Domain). FFTW does NOT need a 1D comm; any comm works.
class FFT3DPlanMPI {
public:
    FFT3DPlanMPI(int nx, int ny, int nz, MPI_Comm comm);
    ~FFT3DPlanMPI();
    FFT3DPlanMPI(const FFT3DPlanMPI&) = delete;
    FFT3DPlanMPI& operator=(const FFT3DPlanMPI&) = delete;

    int nx_global() const { return nx_; }
    int ny_global() const { return ny_; }
    int nz_global() const { return nz_; }

    // FFTW slab extent and start along the outer (z) dim on this rank.
    int local_nz()     const { return static_cast<int>(local_n0_); }
    int local_z_start() const { return static_cast<int>(local_0_start_); }

    // Pad stride along innermost dim for in-place r2c (doubles per i-row).
    int real_row_stride() const { return 2 * (nx_ / 2 + 1); }

    double* real_buf() { return real_buf_; }
    std::complex<Real>* complex_buf() {
        return reinterpret_cast<std::complex<Real>*>(real_buf_);
    }

    void forward();

    MPI_Comm comm() const { return comm_; }

private:
    int nx_, ny_, nz_;
    MPI_Comm comm_;
    long long alloc_local_ = 0, local_n0_ = 0, local_0_start_ = 0;
    double* real_buf_ = nullptr;
    void*   plan_ = nullptr;
};

// Idempotent one-shot fftw_mpi_init.
void init_fftw_mpi();
#endif

}  // namespace blast
