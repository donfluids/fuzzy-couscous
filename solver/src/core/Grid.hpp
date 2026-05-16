#pragma once

#include "core/Types.hpp"

namespace blast {

struct Grid {
    int  nx = 0, ny = 0, nz = 0;
    Real lx = 1.0, ly = 1.0, lz = 1.0;
    Real x0 = 0.0, y0 = 0.0, z0 = 0.0;

    Real dx() const { return lx / nx; }
    Real dy() const { return ly / ny; }
    Real dz() const { return lz / nz; }

    // Cell-centered coordinates (interior index i in [0, nx)).
    Real xc(int i) const { return x0 + (i + Real(0.5)) * dx(); }
    Real yc(int j) const { return y0 + (j + Real(0.5)) * dy(); }
    Real zc(int k) const { return z0 + (k + Real(0.5)) * dz(); }

    Real cell_volume() const { return dx() * dy() * dz(); }
};

}  // namespace blast
