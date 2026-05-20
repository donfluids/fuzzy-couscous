#pragma once

#ifdef BLAST_MPI

#include "core/Grid.hpp"
#include "core/Types.hpp"
#include "parallel/Domain.hpp"

#include <mpi.h>

#include <vector>

namespace blast {

// Describes one rank's local Cartesian sub-block of the global grid.
struct CartDesc {
    int i_off, j_off, k_off;
    int nx, ny, nz;
};

// Describes one rank's z-slab in the FFTW-MPI layout (contiguous z-planes
// spanning the full xy plane).
struct SlabDesc {
    int k_start, k_count;
};

// Allgather every rank's CartDesc / SlabDesc into world-sized vectors so
// the redistribute kernels can compute send/recv intervals without further
// communication. Cost: 2 small Allgather calls.
void gather_cart_descs(const Domain& d, const Grid& global,
                       std::vector<CartDesc>& out);
void gather_slab_descs(MPI_Comm comm, int local_z_start, int local_nz,
                       std::vector<SlabDesc>& out);

// Move data from the solver's 3D-Cartesian layout into the FFTW slab layout.
//
// cart_buf:    rank's local interior, contiguous nx_l*ny_l*nz_l reals,
//              i-fastest (no ghost cells, no padding).
// slab_buf:    rank's local z-slab in FFTW's r2c-padded layout
//              (row_stride = 2*(nx_g/2+1) doubles per i-row).
// nx_g, ny_g:  global extents along x, y.
// row_stride:  padded inner stride in doubles (must equal
//              2*(nx_g/2+1) to match FFTW r2c in-place layout).
// local_nz, local_z_start: this rank's slab extent + start.
void redistribute_cart_to_slab(const double* cart_buf,
                               double* slab_buf,
                               int nx_g, int ny_g,
                               int row_stride,
                               int local_nz, int local_z_start,
                               const std::vector<CartDesc>& cart,
                               const std::vector<SlabDesc>& slab,
                               const Domain& d);

// Reverse: move FFTW slab data back into the rank's 3D-Cartesian local buf.
// cart_buf is overwritten with the values picked from the slab buffer; no
// accumulation (caller adds into Rhs after).
void redistribute_slab_to_cart(const double* slab_buf,
                               double* cart_buf,
                               int nx_g, int ny_g,
                               int row_stride,
                               int local_nz, int local_z_start,
                               const std::vector<CartDesc>& cart,
                               const std::vector<SlabDesc>& slab,
                               const Domain& d);

}  // namespace blast

#endif  // BLAST_MPI
