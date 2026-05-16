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
