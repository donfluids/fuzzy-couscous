#include <gtest/gtest.h>

#include "bc/BC.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "numerics/RHS.hpp"
#include "numerics/RK3.hpp"
#include "physics/EOS.hpp"
#include "physics/MixtureEOS.hpp"
#include "physics/Multifluid.hpp"
#include "physics/ViscousFlux.hpp"

#include <algorithm>
#include <cmath>

using namespace blast;

// ----------------------------------------------------------------------------
// Classic five-equation validation: an isolated material interface (jump in
// volume fraction and density) embedded in a UNIFORM pressure/velocity field,
// advected by the flow. A correctly designed diffuse-interface scheme keeps the
// pressure and velocity uniform to round-off; a naive conservative scheme rings.
// ----------------------------------------------------------------------------

namespace {

struct Result {
    Real max_dp;     // max |p - p0| / p0
    Real max_du;     // max |u - u0| / |u0|
    Real max_dvw;    // max |v|, |w|
    Real a1min, a1max;
    Real rho_consistency;   // max |rho - (Z1+Z2)|
    bool finite;
};

Result run_interface(const MixtureEOS& mix, Real rho1, Real rho2, Real p0,
                     Real u0, int nsteps) {
    Grid g;
    g.nx = 128; g.ny = 4; g.nz = 4;
    g.lx = 1.0; g.ly = g.lx / g.nx * 4; g.lz = g.ly;
    g.x0 = g.y0 = g.z0 = 0.0;

    GammaLaw gl; gl.gamma = 1.4; gl.R = 1.0;
    IdealGas eos{gl};

    State U(g.nx, g.ny, g.nz);
    FiveEqAux aux; aux.allocate(g.nx, g.ny, g.nz, U.ng());

    // Phase-0 slab between xL and xR (volume fraction ~1 inside, ~0 outside),
    // smoothed over a few cells so the central scheme sees a resolved profile.
    const Real xL = 0.3, xR = 0.7, delta = 4.0 * g.dx();
    const Real eps = mix.a_floor;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real x = g.xc(i);
                const Real w = 0.5 * (std::tanh((x - xL) / delta)
                                    - std::tanh((x - xR) / delta));  // ~1 in [xL,xR]
                Real a1 = eps + (1.0 - 2.0 * eps) * w;
                a1 = std::min(std::max(a1, eps), 1.0 - eps);
                const Real Z1 = a1 * rho1, Z2 = (1.0 - a1) * rho2;
                const Real rho = Z1 + Z2;
                const Real rhoe = mix.five_eq_rhoe_from_p(a1, Z1, Z2, p0);
                U[RHO](i,j,k)  = rho;
                U[RHOU](i,j,k) = rho * u0;
                U[RHOV](i,j,k) = 0.0;
                U[RHOW](i,j,k) = 0.0;
                U[RHOE](i,j,k) = rhoe + 0.5 * rho * u0 * u0;
                aux.Z1(i,j,k) = Z1;
                aux.Z2(i,j,k) = Z2;
                aux.a1(i,j,k) = a1;
            }

    BCSet bc;
    bc.xlo = bc.xhi = bc.ylo = bc.yhi = bc.zlo = bc.zhi = BCType::Periodic;

    ViscousParams vp; vp.mu = 0.0;
    vp.rho_floor = 1e-12; vp.eint_floor = 1e-12;

    RK3 driver(g.nx, g.ny, g.nz, U.ng());
    for (int s = 0; s < nsteps; ++s) {
        const Real dt = max_dt_hyperbolic(U, g, eos, 0.4, &aux.a1, &mix, &aux);
        driver.step_5eq(U, aux, g, bc, eos, vp, dt, mix);
    }

    Result r{0,0,0, 1e300,-1e300, 0.0, true};
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real rho = U[RHO](i,j,k);
                const Real u = U[RHOU](i,j,k)/rho;
                const Real v = U[RHOV](i,j,k)/rho;
                const Real w = U[RHOW](i,j,k)/rho;
                const Real ke = 0.5 * rho * (u*u + v*v + w*w);
                Real p, c;
                mix.p_c_5eq(aux.a1(i,j,k), aux.Z1(i,j,k), aux.Z2(i,j,k),
                            rho, U[RHOE](i,j,k) - ke, p, c);
                if (!std::isfinite(p) || !std::isfinite(rho)) r.finite = false;
                r.max_dp = std::max(r.max_dp, std::fabs(p - p0) / p0);
                r.max_du = std::max(r.max_du, std::fabs(u - u0) / std::fabs(u0));
                r.max_dvw = std::max({r.max_dvw, std::fabs(v), std::fabs(w)});
                r.a1min = std::min(r.a1min, aux.a1(i,j,k));
                r.a1max = std::max(r.a1max, aux.a1(i,j,k));
                r.rho_consistency = std::max(r.rho_consistency,
                    std::fabs(rho - (aux.Z1(i,j,k) + aux.Z2(i,j,k))));
            }
    return r;
}

MixtureEOS sg_sg(Real g1, Real pinf1, Real g2, Real pinf2) {
    MixtureEOS m;
    m.mode = MixMode::FiveEquation;
    m.phase[0].kind = PhaseEOS::StiffenedGas; m.phase[0].sg = {g1, pinf1};
    m.phase[1].kind = PhaseEOS::StiffenedGas; m.phase[1].sg = {g2, pinf2};
    return m;
}

}  // namespace

// Two ideal gases (pinf=0), different gamma: pressure & velocity stay uniform to
// round-off as the contact advects. This is the make-or-break property.
TEST(FiveEqInterface, TwoGammaUniformPV) {
    const MixtureEOS m = sg_sg(1.4, 0.0, 1.6, 0.0);
    const Result r = run_interface(m, /*rho1*/ 1.0, /*rho2*/ 10.0,
                                   /*p0*/ 1.0, /*u0*/ 1.0, /*nsteps*/ 60);
    EXPECT_TRUE(r.finite);
    EXPECT_LT(r.max_dp, 1e-9);
    EXPECT_LT(r.max_du, 1e-9);
    EXPECT_LT(r.max_dvw, 1e-9);
    EXPECT_LT(r.rho_consistency, 1e-11);
    EXPECT_GE(r.a1min, -1e-12);
    EXPECT_LE(r.a1max, 1.0 + 1e-12);
}

// Stiffened gas (gas + stiffened liquid, pinf != 0). S and Pi are linear in
// alpha, so pressure stays uniform to round-off too.
TEST(FiveEqInterface, StiffenedGasUniformPV) {
    const MixtureEOS m = sg_sg(1.4, 0.0, 4.4, 6.0);
    const Result r = run_interface(m, /*rho1*/ 1.0, /*rho2*/ 100.0,
                                   /*p0*/ 5.0, /*u0*/ 1.0, /*nsteps*/ 60);
    EXPECT_TRUE(r.finite);
    EXPECT_LT(r.max_dp, 1e-9);
    EXPECT_LT(r.max_du, 1e-9);
    EXPECT_LT(r.rho_consistency, 1e-10);
    EXPECT_GE(r.a1min, -1e-12);
    EXPECT_LE(r.a1max, 1.0 + 1e-12);
}

// A JWL phase (real-products EOS) is non-linear in the phase density, so the
// interface is not exactly oscillation-free, but it must stay well-behaved:
// small pressure error, bounded volume fraction, finite, mass-consistent.
TEST(FiveEqInterface, JWLPhaseStaysWellBehaved) {
    MixtureEOS m;
    m.mode = MixMode::FiveEquation;
    m.phase[0].kind = PhaseEOS::StiffenedGas; m.phase[0].sg = {1.4, 0.0};
    m.phase[1].kind = PhaseEOS::JWLPhase;
    m.phase[1].jwl.A = 0.5; m.phase[1].jwl.B = 0.1;
    m.phase[1].jwl.R1 = 4.0; m.phase[1].jwl.R2 = 1.0;
    m.phase[1].jwl.omega = 0.4; m.phase[1].jwl.rho0 = 2.0;

    const Result r = run_interface(m, /*rho1*/ 1.0, /*rho2*/ 2.0,
                                   /*p0*/ 1.0, /*u0*/ 1.0, /*nsteps*/ 40);
    EXPECT_TRUE(r.finite);
    EXPECT_LT(r.max_dp, 5e-3);
    EXPECT_LT(r.rho_consistency, 1e-9);
    EXPECT_GE(r.a1min, -1e-12);
    EXPECT_LE(r.a1max, 1.0 + 1e-12);
}
