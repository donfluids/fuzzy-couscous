#ifdef BLAST_MPI

#include "parallel/Domain.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace blast {

namespace {

bool axis_periodic(const BCSet& bc, int dim) {
    if (dim == 0) return bc.xlo == BCType::Periodic && bc.xhi == BCType::Periodic;
    if (dim == 1) return bc.ylo == BCType::Periodic && bc.yhi == BCType::Periodic;
    return bc.zlo == BCType::Periodic && bc.zhi == BCType::Periodic;
}

// BLAST_DIMS=PX,PY,PZ overrides MPI_Dims_create. Returns true if set and
// the product matches world_size; false otherwise (caller falls back).
bool parse_dims_env(int world_size, int dims_out[3]) {
    const char* s = std::getenv("BLAST_DIMS");
    if (s == nullptr || *s == 0) return false;
    int px = 0, py = 0, pz = 0;
    if (std::sscanf(s, "%d,%d,%d", &px, &py, &pz) != 3) return false;
    if (px <= 0 || py <= 0 || pz <= 0) return false;
    if (px * py * pz != world_size) return false;
    dims_out[0] = px; dims_out[1] = py; dims_out[2] = pz;
    return true;
}

}  // namespace

Domain::Domain(MPI_Comm world, const Grid& global_grid, const BCSet& bc) {
    int world_size = 0;
    MPI_Comm_size(world, &world_size);

    int periods[3] = {axis_periodic(bc, 0) ? 1 : 0,
                      axis_periodic(bc, 1) ? 1 : 0,
                      axis_periodic(bc, 2) ? 1 : 0};
    periodic_[0] = periods[0]; periodic_[1] = periods[1]; periodic_[2] = periods[2];

    int dims[3] = {0, 0, 0};
    if (!parse_dims_env(world_size, dims))
        MPI_Dims_create(world_size, 3, dims);
    dims_[0] = dims[0]; dims_[1] = dims[1]; dims_[2] = dims[2];

    MPI_Cart_create(world, 3, dims, periods, /*reorder=*/1, &comm_);
    MPI_Comm_rank(comm_, &rank_);
    size_ = world_size;

    int coords[3];
    MPI_Cart_coords(comm_, rank_, 3, coords);
    coords_[0] = coords[0]; coords_[1] = coords[1]; coords_[2] = coords[2];

    for (int d = 0; d < 3; ++d) {
        MPI_Cart_shift(comm_, d, 1, &neighbor_lo_[d], &neighbor_hi_[d]);
    }

    global_extent_[0] = global_grid.nx;
    global_extent_[1] = global_grid.ny;
    global_extent_[2] = global_grid.nz;

    if (rank_ == 0) {
        if (dims_[0] > global_grid.nx || dims_[1] > global_grid.ny || dims_[2] > global_grid.nz) {
            throw std::runtime_error(
                "Domain: more ranks per axis than global cells per axis");
        }
    }

    // Every rank's local extent must be >= NGHOST on each axis, otherwise
    // the halo subarray exchange overlaps with itself and corrupts data
    // silently. Check on all ranks; abort if any rank is below the limit.
    int my_local[3];
    long long off_dummy;
    compute_local_extent_(0, global_grid.nx, my_local[0], off_dummy);
    compute_local_extent_(1, global_grid.ny, my_local[1], off_dummy);
    compute_local_extent_(2, global_grid.nz, my_local[2], off_dummy);
    int min_local[3];
    MPI_Allreduce(my_local, min_local, 3, MPI_INT, MPI_MIN, comm_);
    if (min_local[0] < NGHOST || min_local[1] < NGHOST || min_local[2] < NGHOST) {
        if (rank_ == 0) {
            throw std::runtime_error(
                "Domain: at least one rank has a local extent < NGHOST="
                + std::to_string(NGHOST)
                + " (min per-axis local = (" + std::to_string(min_local[0])
                + ", " + std::to_string(min_local[1])
                + ", " + std::to_string(min_local[2])
                + ")). Use a coarser rank count or a larger global grid.");
        }
        MPI_Barrier(comm_);
        MPI_Abort(comm_, 2);
    }
}

Domain::~Domain() {
    if (comm_ != MPI_COMM_NULL) MPI_Comm_free(&comm_);
}

int Domain::neighbor(int dim, int side) const {
    return side < 0 ? neighbor_lo_[dim] : neighbor_hi_[dim];
}

bool Domain::is_physical_face(int dim, int side) const {
    // Internal partition: there's a real neighbor we communicate with.
    // Physical face: no neighbor (MPI_PROC_NULL) AND not periodic.
    if (periodic_[dim]) return false;        // periodic wrap is handled by halo
    return neighbor(dim, side) == MPI_PROC_NULL;
}

void Domain::compute_local_extent_(int dim, int N_global,
                                   int& n_local, long long& offset) const {
    const int Np = dims_[dim];
    const int p  = coords_[dim];
    const int base = N_global / Np;
    const int rem  = N_global % Np;
    n_local = base + (p < rem ? 1 : 0);
    offset  = static_cast<long long>(base) * p + std::min(p, rem);
}

Grid Domain::local_grid(const Grid& g) const {
    Grid local = g;
    int nl[3];
    long long off[3];
    compute_local_extent_(0, g.nx, nl[0], off[0]);
    compute_local_extent_(1, g.ny, nl[1], off[1]);
    compute_local_extent_(2, g.nz, nl[2], off[2]);

    local.nx = nl[0]; local.ny = nl[1]; local.nz = nl[2];
    local.lx = g.dx() * nl[0];
    local.ly = g.dy() * nl[1];
    local.lz = g.dz() * nl[2];
    local.x0 = g.x0 + g.dx() * off[0];
    local.y0 = g.y0 + g.dy() * off[1];
    local.z0 = g.z0 + g.dz() * off[2];
    return local;
}

std::array<long long, 3> Domain::global_offset(const Grid& g) const {
    std::array<long long, 3> off{};
    int nl;
    compute_local_extent_(0, g.nx, nl, off[0]);
    compute_local_extent_(1, g.ny, nl, off[1]);
    compute_local_extent_(2, g.nz, nl, off[2]);
    return off;
}

}  // namespace blast

#endif  // BLAST_MPI
