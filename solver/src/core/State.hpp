#pragma once

#include "core/Field3D.hpp"
#include "core/Types.hpp"

#include <array>
#include <string>

namespace blast {

// Holds the five conserved fields (rho, rho u, rho v, rho w, rho E) on a
// shared grid + ghost layer.
class State {
public:
    State() = default;
    State(int nx, int ny, int nz, int ng = NGHOST) { allocate(nx, ny, nz, ng); }

    void allocate(int nx, int ny, int nz, int ng = NGHOST) {
        static constexpr const char* names[NCONS] =
            {"rho", "rho_u", "rho_v", "rho_w", "rho_E"};
        for (int v = 0; v < NCONS; ++v) {
            fields_[v].resize(nx, ny, nz, ng);
            fields_[v].set_name(names[v]);
        }
    }

    Field3D&       operator[](int v)       { return fields_[v]; }
    const Field3D& operator[](int v) const { return fields_[v]; }

    int nx() const { return fields_[0].nx(); }
    int ny() const { return fields_[0].ny(); }
    int nz() const { return fields_[0].nz(); }
    int ng() const { return fields_[0].ng(); }

    void fill(int v, Real value) { fields_[v].fill(value); }

private:
    std::array<Field3D, NCONS> fields_;
};

// SAXPY-like operation: out = a*x + b*y, applied across all 5 conserved
// fields, over the interior + ghost region.
inline void state_axpby(State& out, Real a, const State& x, Real b, const State& y) {
    for (int v = 0; v < NCONS; ++v) {
        const Field3D& X = x[v];
        const Field3D& Y = y[v];
        Field3D&       O = out[v];
        const Index N = X.ldx() * (X.ny() + 2 * X.ng()) * (X.nz() + 2 * X.ng());
        const Real* __restrict__ xp = X.raw();
        const Real* __restrict__ yp = Y.raw();
        Real* __restrict__       op = O.raw();
#pragma omp parallel for simd schedule(static)
        for (Index i = 0; i < N; ++i) op[i] = a * xp[i] + b * yp[i];
    }
}

// Positivity floor: clamp density and internal-energy density to small positive
// values so a strong rarefaction / contact overshoot cannot drive a cell to
// negative rho or negative p (which makes c = sqrt(gamma p/rho) NaN and poisons
// the whole field). The internal-energy floor guarantees p > 0 independent of
// the local gamma since e_int = rhoE - ke = p/(gamma-1) = p*G > 0 iff p > 0.
// Only touches cells that would otherwise be invalid, so a run that is already
// positive everywhere is left bit-for-bit unchanged. Returns the number of
// cells clamped (for diagnostics). Interior only; ghosts are reset by BCs.
inline long enforce_positivity(State& U, Real rho_floor, Real eint_floor) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz();
    Field3D& rho = U[RHO];
    Field3D& mx  = U[RHOU];
    Field3D& my  = U[RHOV];
    Field3D& mz  = U[RHOW];
    Field3D& E   = U[RHOE];
    long nclamp = 0;
#pragma omp parallel for collapse(2) schedule(static) reduction(+:nclamp)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                Real r = rho(i,j,k);
                if (!(r >= rho_floor)) { r = rho_floor; rho(i,j,k) = r; ++nclamp; }
                const Real ke = 0.5 * (mx(i,j,k)*mx(i,j,k)
                                     + my(i,j,k)*my(i,j,k)
                                     + mz(i,j,k)*mz(i,j,k)) / r;
                const Real eint = E(i,j,k) - ke;
                if (!(eint >= eint_floor)) { E(i,j,k) = ke + eint_floor; ++nclamp; }
            }
    return nclamp;
}

// out = a*x + b*y + c*z, used by RK3.
inline void state_axpbypcz(State& out, Real a, const State& x, Real b,
                           const State& y, Real c, const State& z) {
    for (int v = 0; v < NCONS; ++v) {
        const Field3D& X = x[v];
        const Field3D& Y = y[v];
        const Field3D& Z = z[v];
        Field3D&       O = out[v];
        const Index N = X.ldx() * (X.ny() + 2 * X.ng()) * (X.nz() + 2 * X.ng());
        const Real* __restrict__ xp = X.raw();
        const Real* __restrict__ yp = Y.raw();
        const Real* __restrict__ zp = Z.raw();
        Real* __restrict__       op = O.raw();
#pragma omp parallel for simd schedule(static)
        for (Index i = 0; i < N; ++i) op[i] = a * xp[i] + b * yp[i] + c * zp[i];
    }
}

}  // namespace blast
