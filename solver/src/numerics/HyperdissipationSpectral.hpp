#pragma once

#include "core/Types.hpp"
#include "physics/ViscousFlux.hpp"

#ifdef BLAST_MPI
#include "parallel/SlabRedistribute.hpp"
#endif

#include <array>
#include <complex>
#include <memory>
#include <vector>

namespace blast {

class FFT3DPlan;
class FFT3DInversePlan;
class R2R3DPlan;
class State;
struct Grid;
#ifdef BLAST_MPI
class Domain;
#endif

// Abstract apply interface so RhsScratch can hold either the serial
// implementation or the MPI one through a single unique_ptr.
class HyperdissipationSpectralBase {
public:
    virtual ~HyperdissipationSpectralBase() = default;
    virtual void apply(const State& U, const Grid& g,
                       Real nu4, Real nu6, State& Rhs) = 0;
};

// Pseudospectral evaluation of the hyperdissipation operator on a periodic
// or all-slip-wall domain. Owns the FFTW plans and working buffers. Single
// rank only.
//
// In spectral space the operator is a per-mode multiplier:
//     RHS_hyper_hat(k) = (-nu4 |k|^4  -  nu6 |k|^6) Uhat(k)
//
// Periodic mode uses complex r2c + c2r (one plan pair, shared by all
// conserved variables; the FFTW round-trip normalization is 1/N).
// SlipWall mode uses per-variable real-to-real DCT-II / DST-II plans:
// the axis carrying the wall-normal momentum component uses DST (odd
// reflection at the wall) and every other axis uses DCT (even reflection).
// The R2R round-trip normalization is 1/(8 N_x N_y N_z) (2N per axis).
//
// All-periodic vs all-slip-wall is validated at config load; mixed BCs
// are not supported here.
class HyperdissipationSpectral : public HyperdissipationSpectralBase {
public:
    HyperdissipationSpectral(int nx, int ny, int nz, SpectralBCMode mode);
    ~HyperdissipationSpectral() override;
    HyperdissipationSpectral(const HyperdissipationSpectral&) = delete;
    HyperdissipationSpectral& operator=(const HyperdissipationSpectral&) = delete;

    void apply(const State& U, const Grid& g, Real nu4, Real nu6,
               State& Rhs) override;

private:
    void apply_periodic_ (const State& U, const Grid& g, Real nu4, Real nu6, State& Rhs);
    void apply_slip_wall_(const State& U, const Grid& g, Real nu4, Real nu6, State& Rhs);

    int nx_, ny_, nz_;
    SpectralBCMode mode_;

    // Periodic mode (unused in SlipWall).
    std::unique_ptr<FFT3DPlan>        dft_fwd_;
    std::unique_ptr<FFT3DInversePlan> dft_inv_;
    std::vector<std::complex<Real>>   spec_buf_;

    // SlipWall mode (one fwd + inv per conserved variable; ρ and ρE
    // happen to share kinds but having one plan per variable simplifies
    // dispatch and the per-plan cost is negligible with FFTW_ESTIMATE).
    std::array<std::unique_ptr<R2R3DPlan>, NCONS> r2r_fwd_;
    std::array<std::unique_ptr<R2R3DPlan>, NCONS> r2r_inv_;

    // Shared real-space working buffer.
    std::vector<Real> real_buf_;
};

#ifdef BLAST_MPI

// MPI sibling of HyperdissipationSpectral. Supports both Periodic and
// SlipWall modes; the BC type is fixed at construction and selects the
// FFTW plan flavour (r2c/c2r vs R2R DCT/DST).
//
// Data flow per RK substage, per conserved variable (both modes):
//   1. Pack U[v] interior on this rank into cart_buf_ (no ghosts, i-fastest)
//   2. Redistribute 3D-Cart -> z-slab (MPI_Alltoallv) into slab_buf_
//   3. Forward FFT (r2c or per-variable R2R, in-place on slab_buf_)
//   4. Multiply each spectral coefficient by (-nu4 |k|^4 - nu6 |k|^6) / norm
//   5. Inverse FFT (c2r or per-variable R2R, in-place on slab_buf_)
//   6. Redistribute z-slab -> 3D-Cart (MPI_Alltoallv) into cart_buf_
//   7. Accumulate cart_buf_ into Rhs[v]
//
// The CartDesc / SlabDesc tables are built once in the ctor (two short
// Allgathers) and reused for every apply() call. All plans operate
// IN-PLACE on slab_buf_ so they can be chained without copying; FFTW's
// new-array execute requires the in-placeness of the call site to match
// the plan, which is why we share a single buffer.
class HyperdissipationSpectralMpi : public HyperdissipationSpectralBase {
public:
    HyperdissipationSpectralMpi(const Grid& global_grid, const Domain& d,
                                SpectralBCMode mode);
    ~HyperdissipationSpectralMpi() override;
    HyperdissipationSpectralMpi(const HyperdissipationSpectralMpi&) = delete;
    HyperdissipationSpectralMpi& operator=(const HyperdissipationSpectralMpi&) = delete;

    void apply(const State& U, const Grid& g, Real nu4, Real nu6,
               State& Rhs) override;

private:
    void apply_periodic_ (const State& U, Real nu4, Real nu6, State& Rhs);
    void apply_slip_wall_(const State& U, Real nu4, Real nu6, State& Rhs);

    const Domain* d_;
    SpectralBCMode mode_;
    int nx_g_, ny_g_, nz_g_;
    Real lx_, ly_, lz_;
    int nx_l_, ny_l_, nz_l_;

    // Plans (Periodic mode: one r2c fwd + one c2r inv; SlipWall mode:
    // per-variable R2R fwd + R2R inv, since each conserved variable has
    // its own kind triple). All operate in-place on slab_buf_.
    void*  fwd_plan_  = nullptr;
    void*  inv_plan_  = nullptr;
    std::array<void*, NCONS> r2r_fwd_plans_ = {};
    std::array<void*, NCONS> r2r_inv_plans_ = {};
    double*  slab_buf_  = nullptr;
    long long local_nz_       = 0;
    long long local_z_start_  = 0;
    int row_stride_           = 0;   // r2c: 2*(nx/2+1); r2r: nx_g

    // Per-axis k^2 tables for SlipWall mode (DCT-II index n -> pi n / L,
    // DST-II index n -> pi (n+1) / L). Built once in ctor.
    std::vector<Real> k2x_dct_, k2x_dst_;
    std::vector<Real> k2y_dct_, k2y_dst_;
    std::vector<Real> k2z_dct_, k2z_dst_;

    std::vector<CartDesc>  cart_descs_;
    std::vector<SlabDesc>  slab_descs_;
    std::vector<double>    cart_buf_;
};

#endif  // BLAST_MPI

}  // namespace blast
