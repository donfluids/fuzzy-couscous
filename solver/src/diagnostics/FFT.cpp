#include "diagnostics/FFT.hpp"

#include <fftw3.h>
#include <omp.h>

#include <mutex>

namespace blast {

namespace {

std::once_flag g_thread_init_flag;

void do_init_fftw_threads() {
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
}

}  // namespace

void init_fftw_threads() {
    std::call_once(g_thread_init_flag, do_init_fftw_threads);
}

FFT3DPlan::FFT3DPlan(int nx, int ny, int nz)
    : nx_(nx), ny_(ny), nz_(nz) {
    init_fftw_threads();
    // FFTW expects row-major with the last dimension fastest, but our
    // physical-space data has i fastest. We label the FFTW dims as
    // (nz, ny, nx) so FFTW's "fastest" matches our "i". Wavenumbers come
    // out indexed analogously.
    auto* in  = fftw_alloc_real(static_cast<std::size_t>(nx_) * ny_ * nz_);
    auto* out = fftw_alloc_complex(complex_size());
    plan_ = static_cast<void*>(fftw_plan_dft_r2c_3d(
        nz_, ny_, nx_, in, out, FFTW_ESTIMATE | FFTW_DESTROY_INPUT));
    fftw_free(in);
    fftw_free(out);
}

FFT3DPlan::~FFT3DPlan() {
    if (plan_) fftw_destroy_plan(static_cast<fftw_plan>(plan_));
}

void FFT3DPlan::forward(const Real* in, std::complex<Real>* out) const {
    fftw_execute_dft_r2c(
        static_cast<fftw_plan>(plan_),
        const_cast<Real*>(in),
        reinterpret_cast<fftw_complex*>(out));
}

}  // namespace blast
