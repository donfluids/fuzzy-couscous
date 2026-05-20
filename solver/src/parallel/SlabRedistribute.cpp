#include "parallel/SlabRedistribute.hpp"

#ifdef BLAST_MPI

#include <algorithm>
#include <cstddef>

namespace blast {

void gather_cart_descs(const Domain& d, const Grid& global,
                       std::vector<CartDesc>& out) {
    Grid lg = d.local_grid(global);
    auto off = d.global_offset(global);
    int my[6] = {
        static_cast<int>(off[0]), static_cast<int>(off[1]),
        static_cast<int>(off[2]),
        lg.nx, lg.ny, lg.nz};
    std::vector<int> buf(static_cast<std::size_t>(d.size()) * 6);
    MPI_Allgather(my, 6, MPI_INT, buf.data(), 6, MPI_INT, d.comm());
    out.resize(static_cast<std::size_t>(d.size()));
    for (int r = 0; r < d.size(); ++r) {
        out[r].i_off = buf[6 * r + 0];
        out[r].j_off = buf[6 * r + 1];
        out[r].k_off = buf[6 * r + 2];
        out[r].nx    = buf[6 * r + 3];
        out[r].ny    = buf[6 * r + 4];
        out[r].nz    = buf[6 * r + 5];
    }
}

void gather_slab_descs(MPI_Comm comm, int local_z_start, int local_nz,
                       std::vector<SlabDesc>& out) {
    int size = 0;
    MPI_Comm_size(comm, &size);
    int my[2] = {local_z_start, local_nz};
    std::vector<int> buf(static_cast<std::size_t>(size) * 2);
    MPI_Allgather(my, 2, MPI_INT, buf.data(), 2, MPI_INT, comm);
    out.resize(static_cast<std::size_t>(size));
    for (int r = 0; r < size; ++r) {
        out[r].k_start = buf[2 * r + 0];
        out[r].k_count = buf[2 * r + 1];
    }
}

void redistribute_cart_to_slab(const double* cart_buf,
                               double* slab_buf,
                               int nx_g, int ny_g,
                               int row_stride,
                               int local_nz, int local_z_start,
                               const std::vector<CartDesc>& cart,
                               const std::vector<SlabDesc>& slab,
                               const Domain& d) {
    const int size = d.size();
    const int my_rank = d.rank();
    const CartDesc& my = cart[my_rank];
    const std::size_t plane_stride =
        static_cast<std::size_t>(ny_g) * row_stride;

    std::vector<int> send_counts(size, 0), send_displs(size, 0);
    std::vector<int> recv_counts(size, 0), recv_displs(size, 0);
    long long total_send = 0, total_recv = 0;

    for (int r = 0; r < size; ++r) {
        const int sl = slab[r].k_start;
        const int sh = sl + slab[r].k_count;
        const int kk_lo = std::max(my.k_off, sl);
        const int kk_hi = std::min(my.k_off + my.nz, sh);
        const int kk_n  = std::max(0, kk_hi - kk_lo);
        send_counts[r] = my.nx * my.ny * kk_n;
        send_displs[r] = static_cast<int>(total_send);
        total_send += send_counts[r];
    }
    for (int s = 0; s < size; ++s) {
        const int kk_lo = std::max(cart[s].k_off, local_z_start);
        const int kk_hi = std::min(cart[s].k_off + cart[s].nz,
                                   local_z_start + local_nz);
        const int kk_n  = std::max(0, kk_hi - kk_lo);
        recv_counts[s] = cart[s].nx * cart[s].ny * kk_n;
        recv_displs[s] = static_cast<int>(total_recv);
        total_recv += recv_counts[s];
    }

    std::vector<double> send_buf(static_cast<std::size_t>(total_send));
    std::vector<double> recv_buf(static_cast<std::size_t>(total_recv));

    for (int r = 0; r < size; ++r) {
        if (send_counts[r] == 0) continue;
        const int sl = slab[r].k_start;
        const int sh = sl + slab[r].k_count;
        const int kk_lo = std::max(my.k_off, sl);
        const int kk_hi = std::min(my.k_off + my.nz, sh);
        double* dst = send_buf.data() + send_displs[r];
        std::size_t p = 0;
        for (int k_g = kk_lo; k_g < kk_hi; ++k_g) {
            const int k_loc = k_g - my.k_off;
            for (int j_loc = 0; j_loc < my.ny; ++j_loc) {
                const std::size_t row_base = static_cast<std::size_t>(my.nx)
                    * (static_cast<std::size_t>(j_loc)
                       + static_cast<std::size_t>(my.ny) * k_loc);
                for (int i_loc = 0; i_loc < my.nx; ++i_loc) {
                    dst[p++] = cart_buf[row_base + i_loc];
                }
            }
        }
    }

    MPI_Alltoallv(send_buf.data(), send_counts.data(), send_displs.data(),
                  MPI_DOUBLE,
                  recv_buf.data(), recv_counts.data(), recv_displs.data(),
                  MPI_DOUBLE,
                  d.comm());

    // Zero the slab buffer (including FFTW's padded i-row tail) so the
    // post-redistribute state is well-defined even at cells not received.
    const std::size_t slab_real_words =
        static_cast<std::size_t>(local_nz) * plane_stride;
    std::fill(slab_buf, slab_buf + slab_real_words, 0.0);

    for (int s = 0; s < size; ++s) {
        if (recv_counts[s] == 0) continue;
        const int kk_lo = std::max(cart[s].k_off, local_z_start);
        const int kk_hi = std::min(cart[s].k_off + cart[s].nz,
                                   local_z_start + local_nz);
        const double* src = recv_buf.data() + recv_displs[s];
        std::size_t p = 0;
        for (int k_g = kk_lo; k_g < kk_hi; ++k_g) {
            const int k_loc = k_g - local_z_start;
            for (int j_loc = 0; j_loc < cart[s].ny; ++j_loc) {
                const int j_g = cart[s].j_off + j_loc;
                const std::size_t row_base =
                    static_cast<std::size_t>(row_stride) * j_g
                    + plane_stride * k_loc;
                for (int i_loc = 0; i_loc < cart[s].nx; ++i_loc) {
                    const int i_g = cart[s].i_off + i_loc;
                    slab_buf[row_base + i_g] = src[p++];
                }
            }
        }
    }
}

void redistribute_slab_to_cart(const double* slab_buf,
                               double* cart_buf,
                               int nx_g, int ny_g,
                               int row_stride,
                               int local_nz, int local_z_start,
                               const std::vector<CartDesc>& cart,
                               const std::vector<SlabDesc>& slab,
                               const Domain& d) {
    (void)nx_g;
    const int size = d.size();
    const int my_rank = d.rank();
    const CartDesc& my = cart[my_rank];
    const std::size_t plane_stride =
        static_cast<std::size_t>(ny_g) * row_stride;

    // Swap send / recv vs the forward direction: this rank's slab is the
    // source, and the Cartesian sub-blocks (one per rank) are the targets.
    std::vector<int> send_counts(size, 0), send_displs(size, 0);
    std::vector<int> recv_counts(size, 0), recv_displs(size, 0);
    long long total_send = 0, total_recv = 0;

    for (int s = 0; s < size; ++s) {
        const int kk_lo = std::max(cart[s].k_off, local_z_start);
        const int kk_hi = std::min(cart[s].k_off + cart[s].nz,
                                   local_z_start + local_nz);
        const int kk_n  = std::max(0, kk_hi - kk_lo);
        send_counts[s] = cart[s].nx * cart[s].ny * kk_n;
        send_displs[s] = static_cast<int>(total_send);
        total_send += send_counts[s];
    }
    for (int r = 0; r < size; ++r) {
        const int sl = slab[r].k_start;
        const int sh = sl + slab[r].k_count;
        const int kk_lo = std::max(my.k_off, sl);
        const int kk_hi = std::min(my.k_off + my.nz, sh);
        const int kk_n  = std::max(0, kk_hi - kk_lo);
        recv_counts[r] = my.nx * my.ny * kk_n;
        recv_displs[r] = static_cast<int>(total_recv);
        total_recv += recv_counts[r];
    }

    std::vector<double> send_buf(static_cast<std::size_t>(total_send));
    std::vector<double> recv_buf(static_cast<std::size_t>(total_recv));

    // Pack: walk each destination rank's Cartesian sub-block on the
    // portion overlapping my slab.
    for (int s = 0; s < size; ++s) {
        if (send_counts[s] == 0) continue;
        const int kk_lo = std::max(cart[s].k_off, local_z_start);
        const int kk_hi = std::min(cart[s].k_off + cart[s].nz,
                                   local_z_start + local_nz);
        double* dst = send_buf.data() + send_displs[s];
        std::size_t p = 0;
        for (int k_g = kk_lo; k_g < kk_hi; ++k_g) {
            const int k_loc = k_g - local_z_start;
            for (int j_loc = 0; j_loc < cart[s].ny; ++j_loc) {
                const int j_g = cart[s].j_off + j_loc;
                const std::size_t row_base =
                    static_cast<std::size_t>(row_stride) * j_g
                    + plane_stride * k_loc;
                for (int i_loc = 0; i_loc < cart[s].nx; ++i_loc) {
                    const int i_g = cart[s].i_off + i_loc;
                    dst[p++] = slab_buf[row_base + i_g];
                }
            }
        }
    }

    MPI_Alltoallv(send_buf.data(), send_counts.data(), send_displs.data(),
                  MPI_DOUBLE,
                  recv_buf.data(), recv_counts.data(), recv_displs.data(),
                  MPI_DOUBLE,
                  d.comm());

    // Unpack into the rank's Cartesian buffer in i-fastest layout.
    for (int r = 0; r < size; ++r) {
        if (recv_counts[r] == 0) continue;
        const int sl = slab[r].k_start;
        const int sh = sl + slab[r].k_count;
        const int kk_lo = std::max(my.k_off, sl);
        const int kk_hi = std::min(my.k_off + my.nz, sh);
        const double* src = recv_buf.data() + recv_displs[r];
        std::size_t p = 0;
        for (int k_g = kk_lo; k_g < kk_hi; ++k_g) {
            const int k_loc = k_g - my.k_off;
            for (int j_loc = 0; j_loc < my.ny; ++j_loc) {
                const std::size_t row_base = static_cast<std::size_t>(my.nx)
                    * (static_cast<std::size_t>(j_loc)
                       + static_cast<std::size_t>(my.ny) * k_loc);
                for (int i_loc = 0; i_loc < my.nx; ++i_loc) {
                    cart_buf[row_base + i_loc] = src[p++];
                }
            }
        }
    }
}

}  // namespace blast

#endif  // BLAST_MPI
