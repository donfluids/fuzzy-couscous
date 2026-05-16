#pragma once

#include "core/Types.hpp"

#include <cstddef>
#include <cstdlib>
#include <string>

namespace blast {

// Aligned, padded 3D scalar field. Storage order: i fastest, k slowest.
// Index range with ghosts: i in [-ng, nx+ng), same for j, k.
// Inner-region loop should be the unit-stride i loop for SIMD.
class Field3D {
public:
    Field3D() = default;
    Field3D(int nx, int ny, int nz, int ng = NGHOST, std::string name = {});
    ~Field3D();

    Field3D(const Field3D&) = delete;
    Field3D& operator=(const Field3D&) = delete;
    Field3D(Field3D&& other) noexcept;
    Field3D& operator=(Field3D&& other) noexcept;

    void resize(int nx, int ny, int nz, int ng = NGHOST);
    void fill(Real value);
    void swap(Field3D& other) noexcept;

    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int nz() const { return nz_; }
    int ng() const { return ng_; }

    // Padded leading dimensions (include ghost cells + alignment pad on i).
    Index ldx() const { return ldx_; }
    Index ldxy() const { return ldx_ * ldy_; }

    std::size_t total_bytes() const { return total_cells_ * sizeof(Real); }

    // Pointer to interior cell (0,0,0). data_ points to the ghost origin.
    Real*       interior() { return data_ + ng_ * (1 + ldx_ + ldxy()); }
    const Real* interior() const { return data_ + ng_ * (1 + ldx_ + ldxy()); }

    // Raw ghost-origin pointer (i,j,k=-ng).
    Real*       raw() { return data_; }
    const Real* raw() const { return data_; }

    // (i,j,k) with i,j,k in [-ng, n+ng). Operator() does the indexing.
    inline Real& operator()(int i, int j, int k) {
        return data_[idx_(i, j, k)];
    }
    inline const Real& operator()(int i, int j, int k) const {
        return data_[idx_(i, j, k)];
    }

    inline Index idx_(int i, int j, int k) const {
        return (i + ng_) + ldx_ * ((j + ng_) + ldy_ * (k + ng_));
    }

    const std::string& name() const { return name_; }
    void set_name(std::string n) { name_ = std::move(n); }

private:
    void allocate_();
    void first_touch_();

    Real*       data_ = nullptr;
    int         nx_ = 0, ny_ = 0, nz_ = 0, ng_ = 0;
    Index       ldx_ = 0, ldy_ = 0, ldz_ = 0;
    std::size_t total_cells_ = 0;
    std::string name_;
};

}  // namespace blast
