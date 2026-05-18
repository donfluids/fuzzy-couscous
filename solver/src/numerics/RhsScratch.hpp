#pragma once

#include "core/Field3D.hpp"
#include "core/State.hpp"
#include "numerics/Gradients.hpp"

namespace blast {

// Per-step scratch buffers used by the RHS pipeline. The previous
// implementation allocated ~30 Field3D objects on EACH RHS call (3 per
// RK3 step), all sized to ~17 MB at 256^3. That generated ~1.5 GB of
// allocator + first-touch traffic per simulation step on the production
// grid, which the OS handles via page faults and zero-fill, all of which
// is wasted bandwidth on a memory-bound code.
//
// RK3 owns one RhsScratch (allocated at construction time) and threads
// it into every RHS function. All fields are sized for the local grid
// once; subsequent calls touch the same physical pages.
struct RhsScratch {
    // Inviscid path
    Field3D theta;              // Ducros sensor on interior + 1 ghost
    Field3D alpha;              // local max-eigenvalue per cell
    Field3D theta_dil;          // sensor dilated along the active direction
    Field3D dilate_tmp;         // working buffer for dilate_sensor_along
    State   Flux_inv;           // inviscid flux scratch (one per direction reuse)

    // Viscous path
    Field3D prim_u, prim_v, prim_w, prim_T;
    CellGradients G;
    State   Flux_visc;

    // Hyperdissipation
    Field3D lap;

    // Allocate every buffer for a grid of given local extent + ghost count.
    // Call once per RK3 driver lifetime.
    void allocate(int nx, int ny, int nz, int ng);
};

}  // namespace blast
