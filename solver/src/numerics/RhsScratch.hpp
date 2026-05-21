#pragma once

#include "core/Field3D.hpp"
#include "core/State.hpp"
#include "numerics/Gradients.hpp"

#include <memory>

namespace blast {

class HyperdissipationSpectralBase;

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
    Field3D contact;            // multifluid: 1 near a gamma-contact (G jump)
    Field3D contact_dil;        // contact flag dilated along the active direction
    State   Flux_inv;           // inviscid flux scratch (one per direction reuse)

    // Viscous path
    Field3D prim_u, prim_v, prim_w, prim_T;
    CellGradients G;
    State   Flux_visc;

    // Localized artificial diffusivity (LAD). Allocated lazily on first use
    // (only when ViscousParams::abv_enabled), so default runs pay nothing.
    Field3D lad_theta;          // div u source (filled on [-3, n+3))
    Field3D lad_strain;         // |S| source   (filled on [-3, n+3))
    Field3D mu_art, beta_art, kappa_art;   // LAD coefficients on [-1, n+1)
    bool    abv_allocated = false;
    Real    abv_nu_max    = 0.0;   // max effective LAD diffusivity (for CFL)

    // When true, the Ducros/WENO sensor is zeroed so the central scheme runs
    // everywhere (LAD-only shock treatment). Set per step by the RK3 driver.
    bool    disable_weno  = false;

    // Hyperdissipation. `lap` holds the first composed Laplacian; `lap2`
    // is the second intermediate for the nabla^6 path and is unused when
    // hyper6_coeff == 0.
    Field3D lap;
    Field3D lap2;

    // Pseudospectral hyperdissipation plan + buffers. Either the serial
    // or MPI implementation (chosen by RK3::init_spectral_hyper_mpi or by
    // lazy serial construction in add_rhs_viscous).
    std::unique_ptr<HyperdissipationSpectralBase> spectral_hyper;

    // Allocate every buffer for a grid of given local extent + ghost count.
    // Call once per RK3 driver lifetime.
    void allocate(int nx, int ny, int nz, int ng);

    // Lazily allocate the LAD buffers (idempotent). Called from add_rhs_viscous
    // the first time artificial diffusivity is requested.
    void allocate_abv(int nx, int ny, int nz, int ng);

    // Out-of-line dtor so the unique_ptr<HyperdissipationSpectral> can be
    // declared with a forward-declared payload.
    RhsScratch();
    ~RhsScratch();
    RhsScratch(const RhsScratch&) = delete;
    RhsScratch& operator=(const RhsScratch&) = delete;
};

}  // namespace blast
