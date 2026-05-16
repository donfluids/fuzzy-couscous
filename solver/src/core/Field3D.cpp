#include "core/Field3D.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>

#include <omp.h>

namespace blast {

namespace {

// Align leading dimension so each (j,k) row starts on a 64B boundary.
constexpr std::size_t kAlignBytes = 64;
constexpr std::size_t kAlignReals = kAlignBytes / sizeof(Real);

Index pad_to_align(Index n) {
    Index r = n % static_cast<Index>(kAlignReals);
    return r == 0 ? n : n + (static_cast<Index>(kAlignReals) - r);
}

}  // namespace

Field3D::Field3D(int nx, int ny, int nz, int ng, std::string name)
    : name_(std::move(name)) {
    resize(nx, ny, nz, ng);
}

Field3D::~Field3D() {
    if (data_) std::free(data_);
}

Field3D::Field3D(Field3D&& other) noexcept { swap(other); }

Field3D& Field3D::operator=(Field3D&& other) noexcept {
    if (this != &other) {
        if (data_) std::free(data_);
        data_ = nullptr;
        swap(other);
    }
    return *this;
}

void Field3D::swap(Field3D& other) noexcept {
    using std::swap;
    swap(data_, other.data_);
    swap(nx_, other.nx_);
    swap(ny_, other.ny_);
    swap(nz_, other.nz_);
    swap(ng_, other.ng_);
    swap(ldx_, other.ldx_);
    swap(ldy_, other.ldy_);
    swap(ldz_, other.ldz_);
    swap(total_cells_, other.total_cells_);
    swap(name_, other.name_);
}

void Field3D::resize(int nx, int ny, int nz, int ng) {
    if (nx <= 0 || ny <= 0 || nz <= 0 || ng < 0) {
        throw std::invalid_argument("Field3D::resize: non-positive extents");
    }
    nx_ = nx; ny_ = ny; nz_ = nz; ng_ = ng;
    ldx_ = pad_to_align(nx + 2 * ng);
    ldy_ = ny + 2 * ng;
    ldz_ = nz + 2 * ng;
    total_cells_ = static_cast<std::size_t>(ldx_) * ldy_ * ldz_;
    if (data_) { std::free(data_); data_ = nullptr; }
    allocate_();
    first_touch_();
}

void Field3D::allocate_() {
    void* p = nullptr;
    if (posix_memalign(&p, kAlignBytes, total_bytes()) != 0) {
        throw std::bad_alloc();
    }
    data_ = static_cast<Real*>(p);
}

// First-touch: zero memory in the same access pattern compute will use,
// so each thread's pages land in its local NUMA node.
void Field3D::first_touch_() {
    const Index ldxy = ldx_ * ldy_;
#pragma omp parallel for schedule(static)
    for (Index k = 0; k < ldz_; ++k) {
        Real* slab = data_ + k * ldxy;
        std::memset(slab, 0, static_cast<std::size_t>(ldxy) * sizeof(Real));
    }
}

void Field3D::fill(Real value) {
    const Index ldxy = ldx_ * ldy_;
#pragma omp parallel for schedule(static)
    for (Index k = 0; k < ldz_; ++k) {
        Real* slab = data_ + k * ldxy;
        for (Index i = 0; i < ldxy; ++i) slab[i] = value;
    }
}

}  // namespace blast
