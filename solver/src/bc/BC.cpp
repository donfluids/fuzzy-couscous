#include "bc/BC.hpp"

#include "core/Field3D.hpp"

#include <omp.h>

namespace blast {

namespace {

// Apply BC on one face for one conserved variable.
// dim: 0=x,1=y,2=z; side: -1=lo,+1=hi; sign_flip: true to flip sign on mirror.
void apply_face(Field3D& F, int dim, int side, BCType type, bool sign_flip) {
    const int nx = F.nx(), ny = F.ny(), nz = F.nz(), ng = F.ng();

    auto mirror_index = [&](int i, int n, int s) {
        // For side=-1, ghost index i in [-ng,-1] maps to interior 2*(-i)-1
        // i.e. i=-1 -> 0, i=-2 -> 1, etc.  For periodic: wrap.
        // For SlipWall: same mirror.
        if (s < 0) return -1 - i;
        return 2 * n - 1 - i;
    };

    const Real sgn = sign_flip ? -1.0 : 1.0;

    if (dim == 0) {
        const int i_start = (side < 0) ? -ng : nx;
        const int i_end   = (side < 0) ?  0  : nx + ng;
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = -ng; k < nz + ng; ++k)
            for (int j = -ng; j < ny + ng; ++j)
                for (int i = i_start; i < i_end; ++i) {
                    int ii;
                    Real s = 1.0;
                    if (type == BCType::Periodic) {
                        ii = (i + nx) % nx;
                    } else if (type == BCType::Outflow) {
                        ii = (side < 0) ? 0 : nx - 1;
                    } else {  // SlipWall
                        ii = mirror_index(i, nx, side);
                        s  = sgn;
                    }
                    F(i, j, k) = s * F(ii, j, k);
                }
    } else if (dim == 1) {
        const int j_start = (side < 0) ? -ng : ny;
        const int j_end   = (side < 0) ?  0  : ny + ng;
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = -ng; k < nz + ng; ++k)
            for (int j = j_start; j < j_end; ++j)
                for (int i = -ng; i < nx + ng; ++i) {
                    int jj;
                    Real s = 1.0;
                    if (type == BCType::Periodic) {
                        jj = (j + ny) % ny;
                    } else if (type == BCType::Outflow) {
                        jj = (side < 0) ? 0 : ny - 1;
                    } else {
                        jj = mirror_index(j, ny, side);
                        s  = sgn;
                    }
                    F(i, j, k) = s * F(i, jj, k);
                }
    } else {  // dim == 2
        const int k_start = (side < 0) ? -ng : nz;
        const int k_end   = (side < 0) ?  0  : nz + ng;
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = k_start; k < k_end; ++k)
            for (int j = -ng; j < ny + ng; ++j)
                for (int i = -ng; i < nx + ng; ++i) {
                    int kk;
                    Real s = 1.0;
                    if (type == BCType::Periodic) {
                        kk = (k + nz) % nz;
                    } else if (type == BCType::Outflow) {
                        kk = (side < 0) ? 0 : nz - 1;
                    } else {
                        kk = mirror_index(k, nz, side);
                        s  = sgn;
                    }
                    F(i, j, k) = s * F(i, j, kk);
                }
    }
}

void apply_face_all_vars(State& U, int dim, int side, BCType type) {
    // Slip wall flips the normal momentum, leaves tangentials and scalars even.
    for (int v = 0; v < NCONS; ++v) {
        bool flip = false;
        if (type == BCType::SlipWall) {
            if      (dim == 0 && v == RHOU) flip = true;
            else if (dim == 1 && v == RHOV) flip = true;
            else if (dim == 2 && v == RHOW) flip = true;
        }
        apply_face(U[v], dim, side, type, flip);
    }
}

}  // namespace

void apply_bcs(State& U, const BCSet& bc) {
    apply_face_all_vars(U, 0, -1, bc.xlo);
    apply_face_all_vars(U, 0, +1, bc.xhi);
    apply_face_all_vars(U, 1, -1, bc.ylo);
    apply_face_all_vars(U, 1, +1, bc.yhi);
    apply_face_all_vars(U, 2, -1, bc.zlo);
    apply_face_all_vars(U, 2, +1, bc.zhi);
}

#ifdef BLAST_MPI
void apply_bcs(State& U, const BCSet& bc, const Domain& d) {
    if (d.is_physical_face(0, -1)) apply_face_all_vars(U, 0, -1, bc.xlo);
    if (d.is_physical_face(0, +1)) apply_face_all_vars(U, 0, +1, bc.xhi);
    if (d.is_physical_face(1, -1)) apply_face_all_vars(U, 1, -1, bc.ylo);
    if (d.is_physical_face(1, +1)) apply_face_all_vars(U, 1, +1, bc.yhi);
    if (d.is_physical_face(2, -1)) apply_face_all_vars(U, 2, -1, bc.zlo);
    if (d.is_physical_face(2, +1)) apply_face_all_vars(U, 2, +1, bc.zhi);
}

void apply_bcs(Field3D& f, const BCSet& bc, const Domain& d) {
    // Scalar (no sign flip). SlipWall -> even mirror = zero-gradient, matching
    // the multifluid fill_G_bcs convention; Periodic wraps; Outflow extrapolates.
    if (d.is_physical_face(0, -1)) apply_face(f, 0, -1, bc.xlo, /*sign_flip=*/false);
    if (d.is_physical_face(0, +1)) apply_face(f, 0, +1, bc.xhi, false);
    if (d.is_physical_face(1, -1)) apply_face(f, 1, -1, bc.ylo, false);
    if (d.is_physical_face(1, +1)) apply_face(f, 1, +1, bc.yhi, false);
    if (d.is_physical_face(2, -1)) apply_face(f, 2, -1, bc.zlo, false);
    if (d.is_physical_face(2, +1)) apply_face(f, 2, +1, bc.zhi, false);
}
#endif

}  // namespace blast
