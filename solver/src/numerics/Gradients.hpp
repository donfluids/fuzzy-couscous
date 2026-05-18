#pragma once

#include "core/Field3D.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "physics/EOS.hpp"

#include <array>

namespace blast {

// 6th-order central first derivatives of (u, v, w, T) at every cell where
// the radius-3 stencil is fully populated. With NGHOST=6, gradients are
// available on interior + 3 ghost layers, which is what the outer-flux
// 6th-order operator subsequently needs.
//
// Layout: gradients[v][d] is the d-th spatial derivative of velocity
// component v (v = 0..2 for u, v, w). dTdx is the temperature gradient.
struct CellGradients {
    std::array<std::array<Field3D, 3>, 3> du;   // du[v][d] = d u_v / d x_d
    std::array<Field3D, 3>                dT;   // dT[d]    = d T   / d x_d

    void allocate(int nx, int ny, int nz, int ng);
};

// Two overloads: the bare one allocates the primitive scratch on the stack
// (4 Field3D objects, ~70 MB at 256^3). The scratch-aware overload takes
// pre-allocated buffers (typically owned by RhsScratch via RK3) so the
// allocation only happens once per simulation lifetime.
void compute_cell_gradients(const State& U, const Grid& g, const IdealGas& eos,
                            CellGradients& G);
void compute_cell_gradients(const State& U, const Grid& g, const IdealGas& eos,
                            Field3D& prim_u, Field3D& prim_v,
                            Field3D& prim_w, Field3D& prim_T,
                            CellGradients& G);

}  // namespace blast
