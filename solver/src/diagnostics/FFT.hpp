#pragma once

#include "core/Types.hpp"

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

}  // namespace blast
