#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace blast {

using Real = double;
using Index = std::int64_t;

// Ghost-layer width. Must be >= max stencil radius used in chained
// operators. WENO5 needs 3; composed first-derivative viscous fluxes
// (two 6th-order stencils in sequence) need 6.
inline constexpr int NGHOST = 6;
inline constexpr int NCONS = 5;

enum ConsVar : int { RHO = 0, RHOU = 1, RHOV = 2, RHOW = 3, RHOE = 4 };

struct GammaLaw {
    Real gamma = 1.4;
    Real R     = 287.05;
    Real cv() const { return R / (gamma - 1.0); }
    Real cp() const { return gamma * R / (gamma - 1.0); }
};

}  // namespace blast
