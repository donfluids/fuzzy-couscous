#ifdef BLAST_MPI

#include "parallel/Halo.hpp"

#include "core/Field3D.hpp"

#include <array>
#include <vector>

namespace blast {

Halo::Halo(const State& U, const Domain& d)
    : domain_(d), nx_(U.nx()), ny_(U.ny()), nz_(U.nz()), ng_(U.ng()) {
    build_datatypes_(U[0]);
}

Halo::~Halo() {
    for (auto& t : slab_type_) {
        if (t != MPI_DATATYPE_NULL) MPI_Type_free(&t);
    }
}

void Halo::build_datatypes_(const Field3D& f) {
    // Field3D padded array shape, row-major (k slowest, i fastest):
    //   sizes  = (ldz, ldy, ldx)        ldz = nz + 2 ng, ldy = ny + 2 ng,
    //                                   ldx padded to alignment
    // Pick a strided subarray for each face: ng layers in one dim, full
    // extent (including ghosts) in the other two. This way the same
    // datatype works for sends and receives -- only the buffer pointer
    // changes.
    const long long ldx = f.ldx();
    const long long ldy = ny_ + 2 * ng_;
    const long long ldz = nz_ + 2 * ng_;

    int sizes[3] = {static_cast<int>(ldz), static_cast<int>(ldy), static_cast<int>(ldx)};

    // dim 0 (x-faces): subarray is ng wide in x, full ldy * ldz in the others.
    {
        int subsizes[3] = {static_cast<int>(ldz), static_cast<int>(ldy), ng_};
        int starts[3]   = {0, 0, 0};
        MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                                 MPI_DOUBLE, &slab_type_[0]);
        MPI_Type_commit(&slab_type_[0]);
    }
    // dim 1 (y-faces): subarray is ng tall in y, full extent in x and z.
    {
        int subsizes[3] = {static_cast<int>(ldz), ng_, static_cast<int>(ldx)};
        int starts[3]   = {0, 0, 0};
        MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                                 MPI_DOUBLE, &slab_type_[1]);
        MPI_Type_commit(&slab_type_[1]);
    }
    // dim 2 (z-faces).
    {
        int subsizes[3] = {ng_, static_cast<int>(ldy), static_cast<int>(ldx)};
        int starts[3]   = {0, 0, 0};
        MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                                 MPI_DOUBLE, &slab_type_[2]);
        MPI_Type_commit(&slab_type_[2]);
    }

    // Offsets in linear-array elements (Real units), measured from
    // f.raw() (which points to ghost-origin = padded (i,j,k) = (-ng, -ng, -ng)).
    // For dim 0:
    //   send_lo: interior slab at i = 0 .. ng-1   -> raw + ng (in x), 0 in yz
    //   send_hi: interior slab at i = nx-ng..nx-1 -> raw + (ng + nx - ng) = raw + nx
    //   recv_lo: ghost slab at i = -ng..-1 -> raw + 0  (origin)
    //   recv_hi: ghost slab at i = nx..nx+ng-1 -> raw + (ng + nx)
    // x is the innermost dim with stride 1. So we add the appropriate column
    // count to the raw-origin pointer.
    send_lo_off_[0] = ng_;
    send_hi_off_[0] = nx_;
    recv_lo_off_[0] = 0;
    recv_hi_off_[0] = ng_ + nx_;

    // For dim 1, stride is ldx, so offsets are in rows.
    send_lo_off_[1] = ldx * ng_;
    send_hi_off_[1] = ldx * ny_;
    recv_lo_off_[1] = 0;
    recv_hi_off_[1] = ldx * (ng_ + ny_);

    // For dim 2, stride is ldx * ldy.
    send_lo_off_[2] = ldx * ldy * ng_;
    send_hi_off_[2] = ldx * ldy * nz_;
    recv_lo_off_[2] = 0;
    recv_hi_off_[2] = ldx * ldy * (ng_ + nz_);
}

void Halo::exchange(State& U) {
    // Post non-blocking sends and receives for all 5 conserved variables
    // and all 3 dims. 20 messages total.
    std::array<MPI_Request, NCONS * 12> reqs;
    int n_reqs = 0;

    for (int v = 0; v < NCONS; ++v) {
        Field3D& F = U[v];
        double* base = F.raw();
        for (int d = 0; d < 3; ++d) {
            const int nb_lo = domain_.neighbor(d, -1);
            const int nb_hi = domain_.neighbor(d, +1);
            const int tag_send_lo = 100 * v + 10 * d + 0;
            const int tag_send_hi = 100 * v + 10 * d + 1;

            if (nb_lo != MPI_PROC_NULL) {
                MPI_Irecv(base + recv_lo_off_[d], 1, slab_type_[d],
                          nb_lo, tag_send_hi, domain_.comm(), &reqs[n_reqs++]);
                MPI_Isend(base + send_lo_off_[d], 1, slab_type_[d],
                          nb_lo, tag_send_lo, domain_.comm(), &reqs[n_reqs++]);
            }
            if (nb_hi != MPI_PROC_NULL) {
                MPI_Irecv(base + recv_hi_off_[d], 1, slab_type_[d],
                          nb_hi, tag_send_lo, domain_.comm(), &reqs[n_reqs++]);
                MPI_Isend(base + send_hi_off_[d], 1, slab_type_[d],
                          nb_hi, tag_send_hi, domain_.comm(), &reqs[n_reqs++]);
            }
        }
    }

    MPI_Waitall(n_reqs, reqs.data(), MPI_STATUSES_IGNORE);
}

}  // namespace blast

#endif  // BLAST_MPI
