#include "numerics/RhsScratch.hpp"

namespace blast {

void RhsScratch::allocate(int nx, int ny, int nz, int ng) {
    theta.resize(nx, ny, nz, ng);
    alpha.resize(nx, ny, nz, ng);
    theta_dil.resize(nx, ny, nz, ng);
    dilate_tmp.resize(nx, ny, nz, ng);
    Flux_inv.allocate(nx, ny, nz, ng);

    prim_u.resize(nx, ny, nz, ng);
    prim_v.resize(nx, ny, nz, ng);
    prim_w.resize(nx, ny, nz, ng);
    prim_T.resize(nx, ny, nz, ng);
    G.allocate(nx, ny, nz, ng);
    Flux_visc.allocate(nx, ny, nz, ng);

    lap.resize(nx, ny, nz, ng);
}

}  // namespace blast
