#include "numerics/Filter.hpp"

#include "core/Field3D.hpp"
#include "numerics/Stencils.hpp"

#include <omp.h>

namespace blast {

namespace {

// Sweep filter along direction d on a single field. Writes filtered values
// into `out`, reads from `in`. Operates over interior only (the radius-3
// stencil needs 3 ghost cells which BCs must have populated).
void filter_sweep(const Field3D& in, Field3D& out, int d, Real sigma) {
    const int nx = in.nx(), ny = in.ny(), nz = in.nz();
    const Index s = (d == 0 ? 1 : (d == 1 ? in.ldx() : in.ldxy()));
    const Real one_minus_sigma = 1.0 - sigma;

#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j) {
#pragma omp simd
            for (int i = 0; i < nx; ++i) {
                const Real* p = &in(i, j, k);
                const Real f = stencil::filter_6(p, s);
                out(i, j, k) = one_minus_sigma * (*p) + sigma * f;
            }
        }
}

}  // namespace

void apply_lele_filter(State& U, const BCSet& bc, Real sigma) {
    if (sigma <= 0.0) return;
    const int nx = U.nx(), ny = U.ny(), nz = U.nz(), ng = U.ng();
    Field3D scratch(nx, ny, nz, ng);

    for (int v = 0; v < NCONS; ++v) {
        Field3D& F = U[v];
        // x sweep -> scratch
        apply_bcs(U, bc);
        filter_sweep(F, scratch, 0, sigma);
        // copy back
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) F(i, j, k) = scratch(i, j, k);

        apply_bcs(U, bc);
        filter_sweep(F, scratch, 1, sigma);
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) F(i, j, k) = scratch(i, j, k);

        apply_bcs(U, bc);
        filter_sweep(F, scratch, 2, sigma);
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) F(i, j, k) = scratch(i, j, k);
    }
}

}  // namespace blast
