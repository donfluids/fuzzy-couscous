#include "diagnostics/FFT.hpp"

#include <fftw3.h>
#include <omp.h>

#ifdef BLAST_MPI
#include <fftw3-mpi.h>
#endif

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

FFT3DInversePlan::FFT3DInversePlan(int nx, int ny, int nz)
    : nx_(nx), ny_(ny), nz_(nz) {
    init_fftw_threads();
    auto* in  = fftw_alloc_complex(complex_size());
    auto* out = fftw_alloc_real(static_cast<std::size_t>(nx_) * ny_ * nz_);
    // Match the dim labelling used in FFT3DPlan: (nz, ny, nx) so FFTW's
    // fastest dim corresponds to our i-fastest layout.
    plan_ = static_cast<void*>(fftw_plan_dft_c2r_3d(
        nz_, ny_, nx_, in, out, FFTW_ESTIMATE | FFTW_DESTROY_INPUT));
    fftw_free(in);
    fftw_free(out);
}

FFT3DInversePlan::~FFT3DInversePlan() {
    if (plan_) fftw_destroy_plan(static_cast<fftw_plan>(plan_));
}

void FFT3DInversePlan::inverse(std::complex<Real>* in, Real* out) const {
    fftw_execute_dft_c2r(
        static_cast<fftw_plan>(plan_),
        reinterpret_cast<fftw_complex*>(in),
        out);
}

// Verify our duplicated r2r kind enums match FFTW's at compile time.
// (The MPI block re-declares the DCT_II/DST_II asserts; keep these for
// builds without BLAST_MPI defined, and add the inverse-kind asserts.)
static_assert(static_cast<int>(r2r::DCT_II)  == FFTW_REDFT10,
              "r2r::DCT_II must equal FFTW_REDFT10");
static_assert(static_cast<int>(r2r::DCT_III) == FFTW_REDFT01,
              "r2r::DCT_III must equal FFTW_REDFT01");
static_assert(static_cast<int>(r2r::DST_II)  == FFTW_RODFT10,
              "r2r::DST_II must equal FFTW_RODFT10");
static_assert(static_cast<int>(r2r::DST_III) == FFTW_RODFT01,
              "r2r::DST_III must equal FFTW_RODFT01");

R2R3DPlan::R2R3DPlan(int nx, int ny, int nz,
                     r2r::Kind kind_z, r2r::Kind kind_y, r2r::Kind kind_x)
    : nx_(nx), ny_(ny), nz_(nz) {
    init_fftw_threads();
    auto* buf = fftw_alloc_real(static_cast<std::size_t>(nx_) * ny_ * nz_);
    const fftw_r2r_kind kinds[3] = {
        static_cast<fftw_r2r_kind>(kind_z),
        static_cast<fftw_r2r_kind>(kind_y),
        static_cast<fftw_r2r_kind>(kind_x),
    };
    plan_ = static_cast<void*>(fftw_plan_r2r_3d(
        nz_, ny_, nx_, buf, buf,
        kinds[0], kinds[1], kinds[2],
        FFTW_ESTIMATE));
    fftw_free(buf);
}

R2R3DPlan::~R2R3DPlan() {
    if (plan_) fftw_destroy_plan(static_cast<fftw_plan>(plan_));
}

void R2R3DPlan::execute(Real* buf) const {
    fftw_execute_r2r(static_cast<fftw_plan>(plan_), buf, buf);
}

#ifdef BLAST_MPI

namespace {
std::once_flag g_fftw_mpi_init_flag;
void do_init_fftw_mpi() {
    fftw_mpi_init();
}
}  // namespace

void init_fftw_mpi() {
    std::call_once(g_fftw_mpi_init_flag, do_init_fftw_mpi);
}

FFT3DPlanMPI::FFT3DPlanMPI(int nx, int ny, int nz, MPI_Comm comm)
    : nx_(nx), ny_(ny), nz_(nz), comm_(comm) {
    init_fftw_threads();
    init_fftw_mpi();
    fftw_plan_with_nthreads(omp_get_max_threads());

    // Slab decomp along the outer (z) dim. FFTW labels dims as (nz, ny, nx);
    // we query alloc size in complex words; in-place r2c needs 2*alloc reals.
    ptrdiff_t local_n0, local_0_start;
    const ptrdiff_t alloc = fftw_mpi_local_size_3d(
        nz_, ny_, nx_ / 2 + 1, comm_, &local_n0, &local_0_start);
    alloc_local_   = static_cast<long long>(alloc);
    local_n0_      = static_cast<long long>(local_n0);
    local_0_start_ = static_cast<long long>(local_0_start);

    real_buf_ = fftw_alloc_real(2 * static_cast<std::size_t>(alloc));

    plan_ = static_cast<void*>(fftw_mpi_plan_dft_r2c_3d(
        nz_, ny_, nx_,
        real_buf_,
        reinterpret_cast<fftw_complex*>(real_buf_),
        comm_,
        FFTW_ESTIMATE));
}

FFT3DPlanMPI::~FFT3DPlanMPI() {
    if (plan_) fftw_destroy_plan(static_cast<fftw_plan>(plan_));
    if (real_buf_) fftw_free(real_buf_);
}

void FFT3DPlanMPI::forward() {
    fftw_mpi_execute_dft_r2c(
        static_cast<fftw_plan>(plan_),
        real_buf_,
        reinterpret_cast<fftw_complex*>(real_buf_));
}

// Verify our duplicated kind enums match FFTW's at compile time.
static_assert(static_cast<int>(r2r::DCT_II) == FFTW_REDFT10,
              "r2r::DCT_II must equal FFTW_REDFT10");
static_assert(static_cast<int>(r2r::DST_II) == FFTW_RODFT10,
              "r2r::DST_II must equal FFTW_RODFT10");

R2R3DPlanMPI::R2R3DPlanMPI(int nx, int ny, int nz, MPI_Comm comm,
                           r2r::Kind kind_z, r2r::Kind kind_y, r2r::Kind kind_x)
    : nx_(nx), ny_(ny), nz_(nz), comm_(comm) {
    init_fftw_threads();
    init_fftw_mpi();
    fftw_plan_with_nthreads(omp_get_max_threads());

    // R2R has no Hermitian fold: full nz x ny x nx reals in, same out.
    // Slab decomp along outer z dim, identical to the r2c case.
    ptrdiff_t local_n0, local_0_start;
    const ptrdiff_t alloc = fftw_mpi_local_size_3d(
        nz_, ny_, nx_, comm_, &local_n0, &local_0_start);
    alloc_local_   = static_cast<long long>(alloc);
    local_n0_      = static_cast<long long>(local_n0);
    local_0_start_ = static_cast<long long>(local_0_start);

    real_buf_ = fftw_alloc_real(static_cast<std::size_t>(alloc));

    const fftw_r2r_kind kinds[3] = {
        static_cast<fftw_r2r_kind>(kind_z),
        static_cast<fftw_r2r_kind>(kind_y),
        static_cast<fftw_r2r_kind>(kind_x),
    };
    plan_ = static_cast<void*>(fftw_mpi_plan_r2r_3d(
        nz_, ny_, nx_,
        real_buf_, real_buf_,
        comm_,
        kinds[0], kinds[1], kinds[2],
        FFTW_ESTIMATE));
}

R2R3DPlanMPI::~R2R3DPlanMPI() {
    if (plan_) fftw_destroy_plan(static_cast<fftw_plan>(plan_));
    if (real_buf_) fftw_free(real_buf_);
}

void R2R3DPlanMPI::forward() {
    fftw_mpi_execute_r2r(
        static_cast<fftw_plan>(plan_), real_buf_, real_buf_);
}

#endif  // BLAST_MPI

}  // namespace blast
