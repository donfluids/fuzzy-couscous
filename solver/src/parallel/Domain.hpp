#pragma once

#ifdef BLAST_MPI

#include "core/Config.hpp"
#include "core/Grid.hpp"
#include "core/Types.hpp"

#include <mpi.h>

#include <array>

namespace blast {

// 3D Cartesian rank topology + global-to-local grid carving.
//
// Construction: pass the GLOBAL grid and the BC set. Factor `world_size`
// into (npx, npy, npz) via MPI_Dims_create, build a Cartesian communicator
// with periodicity per axis matched to BCSet, cache neighbor ranks.
//
// Decomposition rule: cells split as evenly as possible per axis. If the
// global N is not divisible by Np, the first (N % Np) ranks get one extra
// cell.
class Domain {
public:
    Domain(MPI_Comm world, const Grid& global_grid, const BCSet& bc);
    ~Domain();

    Domain(const Domain&) = delete;
    Domain& operator=(const Domain&) = delete;

    MPI_Comm comm() const { return comm_; }
    int rank() const      { return rank_; }
    int size() const      { return size_; }

    const std::array<int, 3>& coords() const { return coords_; }
    const std::array<int, 3>& dims()   const { return dims_; }

    // Neighbor rank on (dim, side). side = -1 (lo) or +1 (hi). Returns
    // MPI_PROC_NULL on non-periodic physical-boundary faces.
    int neighbor(int dim, int side) const;

    // True on the ranks that own a physical-domain face. False on internal
    // partition faces (where halo exchange replaces BC application).
    bool is_physical_face(int dim, int side) const;

    // Carve the local grid for this rank out of the global one.
    Grid local_grid(const Grid& global) const;

    // Global cell-index offset of this rank's interior (i, j, k) origin.
    std::array<long long, 3> global_offset(const Grid& global) const;

    std::array<int, 3> global_extent() const { return global_extent_; }

private:
    void compute_local_extent_(int dim, int N_global,
                               int& n_local, long long& offset) const;

    MPI_Comm comm_ = MPI_COMM_NULL;
    int rank_ = 0;
    int size_ = 1;
    std::array<int, 3> dims_ = {1, 1, 1};
    std::array<int, 3> coords_ = {0, 0, 0};
    std::array<int, 3> neighbor_lo_ = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    std::array<int, 3> neighbor_hi_ = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    std::array<int, 3> global_extent_ = {0, 0, 0};
    std::array<bool, 3> periodic_ = {false, false, false};
};

}  // namespace blast

#endif  // BLAST_MPI
