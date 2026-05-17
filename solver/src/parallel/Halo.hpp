#pragma once

#ifdef BLAST_MPI

#include "core/State.hpp"
#include "parallel/Domain.hpp"

#include <mpi.h>
#include <array>

namespace blast {

// Non-blocking halo exchange for the 6 ghost layers (NGHOST) of every
// conserved variable. Builds MPI subarray datatypes once per Field3D
// shape and reuses them across calls.
//
// Pattern per exchange:
//   - Six pairs of MPI_Irecv (into ghost slabs) + MPI_Isend (from interior
//     slabs adjacent to each face).
//   - One MPI_Waitall.
// Total 30 messages per call (5 conserved variables x 6 faces).
//
// The interior region adjacent to a face is the boundary slab of width
// NGHOST. The ghost region on the opposite face is the receive target.
// For dim 0, lo face: send U(0..NGHOST-1, *, *), recv into U(-NGHOST..-1, *, *).
class Halo {
public:
    Halo(const State& U, const Domain& d);
    ~Halo();

    Halo(const Halo&) = delete;
    Halo& operator=(const Halo&) = delete;

    // Exchange ghost cells. Caller is responsible for applying physical BCs
    // on faces that don't have an MPI neighbor (use Domain::is_physical_face).
    void exchange(State& U);

private:
    void build_datatypes_(const Field3D& f);

    const Domain& domain_;
    int nx_ = 0, ny_ = 0, nz_ = 0, ng_ = 0;

    // For each (dim 0..2): a strided slab type covering NGHOST layers along
    // that dim and full extent in the other two dims (including ghosts so
    // edge / corner cells round-trip across multiple exchanges if needed).
    std::array<MPI_Datatype, 3> slab_type_{};
    // Index offsets within Field3D::raw() for the four slab positions per
    // dim: send_lo, send_hi, recv_lo, recv_hi.
    std::array<long long, 3> send_lo_off_{};
    std::array<long long, 3> send_hi_off_{};
    std::array<long long, 3> recv_lo_off_{};
    std::array<long long, 3> recv_hi_off_{};
};

}  // namespace blast

#endif  // BLAST_MPI
