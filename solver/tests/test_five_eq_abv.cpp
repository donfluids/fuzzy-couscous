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
#include <vector>

using namespace blast;

// ----------------------------------------------------------------------------
// Five-equation model WITH localized artificial diffusivity (LAD). Verifies the
// LAD extension to the aux fields (Z1, Z2, alpha1): the contact/interface
// diffusivity smooths the material interface while the consistent internal-energy
// diffusion keeps the contact in pressure/velocity equilibrium (exact for
// stiffened-gas phases), and shocks crossing the interface stay bounded/positive.
// ----------------------------------------------------------------------------

namespace {

MixtureEOS sg_sg(Real g1, Real pinf1, Real g2, Real pinf2) {
    MixtureEOS m;
    m.mode = MixMode::FiveEquation;
    m.phase[0].kind = PhaseEOS::StiffenedGas; m.phase[0].sg = {g1, pinf1};
    m.phase[1].kind = PhaseEOS::StiffenedGas; m.phase[1].sg = {g2, pinf2};
    return m;
}

struct Result {
    Real max_dp;     // max |p - p0| / p0
    Real max_u_abs;  // max |u - u0|  (absolute)
    Real max_dvw;    // max |v|, |w|
    Real a1min, a1max;
    Real rho_consistency;   // max |rho - (Z1+Z2)|
    Real thickness;  // sum_i a1(1-a1) along centerline (interface thickness)
    bool finite;
};

// Interface thickness proxy: sum of a1*(1-a1) along the centerline. It is ~0
// where a1 is saturated (0 or 1) and peaks in the transition, so it GROWS as the
// contact diffusivity spreads the interface over more cells. Insensitive to the
// even/odd structure of the wide LAD Laplacian (unlike a nearest-neighbor slope).
Real interface_thickness(const FiveEqAux& aux, const Grid& g) {
    const int jc = g.ny / 2, kc = g.nz / 2;
    Real t = 0.0;
    for (int i = 0; i < g.nx; ++i) {
        const Real a = aux.a1(i, jc, kc);
        t += a * (1.0 - a);
    }
    return t;
}

// Advected (or stationary) material slab in a uniform p,u field, run with the
// supplied ViscousParams (LAD config). Optionally snapshots the final interior
// state (rho, mom, E, Z1, Z2, a1) for the no-op comparison.
Result run_contact(const MixtureEOS& mix, Real rho1, Real rho2, Real p0, Real u0,
                   int nsteps, const ViscousParams& vp_in, Real delta_cells = 4.0,
                   std::vector<Real>* snapshot = nullptr) {
    Grid g;
    g.nx = 128; g.ny = 4; g.nz = 4;
    g.lx = 1.0; g.ly = g.lx / g.nx * 4; g.lz = g.ly;
    g.x0 = g.y0 = g.z0 = 0.0;

    GammaLaw gl; gl.gamma = 1.4; gl.R = 1.0;
    IdealGas eos{gl};

    State U(g.nx, g.ny, g.nz);
    FiveEqAux aux; aux.allocate(g.nx, g.ny, g.nz, U.ng());

    const Real xL = 0.3, xR = 0.7, delta = delta_cells * g.dx();
    const Real eps = mix.a_floor;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real x = g.xc(i);
                const Real w = 0.5 * (std::tanh((x - xL) / delta)
                                    - std::tanh((x - xR) / delta));
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

    ViscousParams vp = vp_in;
    vp.rho_floor = 1e-12; vp.eint_floor = 1e-12;

    const Real dxm = std::min({g.dx(), g.dy(), g.dz()});

    RK3 driver(g.nx, g.ny, g.nz, U.ng());
    for (int s = 0; s < nsteps; ++s) {
        Real dt = max_dt_hyperbolic(U, g, eos, 0.4, &aux.a1, &mix, &aux);
        const Real num = driver.last_abv_nu_max();
        if (num > 0.0) dt = std::min(dt, 0.25 * dxm * dxm / num);
        driver.step_5eq(U, aux, g, bc, eos, vp, dt, mix);
    }

    Result r{0,0,0, 1e300,-1e300, 0.0, 0.0, true};
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
                r.max_u_abs = std::max(r.max_u_abs, std::fabs(u - u0));
                r.max_dvw = std::max({r.max_dvw, std::fabs(v), std::fabs(w)});
                r.a1min = std::min(r.a1min, aux.a1(i,j,k));
                r.a1max = std::max(r.a1max, aux.a1(i,j,k));
                r.rho_consistency = std::max(r.rho_consistency,
                    std::fabs(rho - (aux.Z1(i,j,k) + aux.Z2(i,j,k))));
            }
    r.thickness = interface_thickness(aux, g);

    if (snapshot) {
        snapshot->clear();
        snapshot->reserve(static_cast<size_t>(g.nx) * g.ny * g.nz * 6);
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j)
                for (int i = 0; i < g.nx; ++i) {
                    snapshot->push_back(U[RHO](i,j,k));
                    snapshot->push_back(U[RHOU](i,j,k));
                    snapshot->push_back(U[RHOE](i,j,k));
                    snapshot->push_back(aux.Z1(i,j,k));
                    snapshot->push_back(aux.Z2(i,j,k));
                    snapshot->push_back(aux.a1(i,j,k));
                }
    }
    return r;
}

ViscousParams lad_5eq(Real cD, bool weno_on) {
    ViscousParams vp;
    vp.mu = 0.0;
    vp.abv_enabled = true;
    vp.abv_cbeta = 1.0;
    vp.abv_cmu = 0.002;
    vp.abv_ckappa = 0.01;   // ignored by the 5eq path (kappa disabled), set anyway
    vp.abv_cD = cD;
    vp.abv_disable_weno = !weno_on;
    return vp;
}

}  // namespace

// STRONG correctness: a STATIONARY stiffened-gas contact, LAD-only with a large
// contact diffusivity. The consistent internal-energy diffusion must keep the
// pressure uniform to round-off while LAD visibly smooths the interface.
TEST(FiveEqAbv, StationaryContactLADKeepsPressureUniform) {
    const MixtureEOS m = sg_sg(1.4, 0.0, 4.4, 6.0);
    // Sharp (under-resolved) interface: the regime where LAD is actually needed
    // and the scale-selective sensor fires. A stationary uniform-(p,u) contact is
    // steady, so with LAD off the interface keeps its initial sharpness.
    const Real dc = 1.0;  // interface width in cells
    const Result off = run_contact(m, 1.0, 100.0, 5.0, 0.0, 60,
                                   lad_5eq(/*cD*/ 0.0, /*weno_on*/ false), dc);
    const Result on  = run_contact(m, 1.0, 100.0, 5.0, 0.0, 60,
                                   lad_5eq(/*cD*/ 0.5, /*weno_on*/ false), dc);
    EXPECT_TRUE(on.finite);
    EXPECT_LT(on.max_dp, 1e-9) << "LAD on a 5eq contact injected spurious pressure";
    EXPECT_LT(on.max_u_abs, 1e-9);
    EXPECT_LT(on.max_dvw, 1e-9);
    EXPECT_LT(on.rho_consistency, 1e-10);
    EXPECT_GE(on.a1min, -1e-12);
    EXPECT_LE(on.a1max, 1.0 + 1e-12);
    // Prove the contact diffusivity actually regularized the interface (did not
    // just preserve p by doing nothing): it must spread alpha1 over more cells
    // than the (steady) baseline.
    EXPECT_GT(on.thickness, 1.1 * off.thickness)
        << "contact diffusivity did not spread the interface (on=" << on.thickness
        << " off=" << off.thickness << ")";
}

// Moving contact with WENO + LAD on: the consistent internal-energy diffusion
// keeps the contact pressure-preserving under advection (up to the small O(D*du)
// residual) while staying bounded and mass-consistent.
TEST(FiveEqAbv, MovingContactLADPreservesPressure) {
    const MixtureEOS m = sg_sg(1.4, 0.0, 4.4, 6.0);
    const Result on = run_contact(m, 1.0, 100.0, 5.0, 1.0, 60,
                                  lad_5eq(/*cD*/ 0.1, /*weno_on*/ true));
    EXPECT_TRUE(on.finite);
    EXPECT_LT(on.max_dp, 1e-7);
    EXPECT_LT(on.max_dvw, 1e-7);
    EXPECT_LT(on.rho_consistency, 1e-9);
    EXPECT_GE(on.a1min, -1e-12);
    EXPECT_LE(on.a1max, 1.0 + 1e-12);
}

// abv_enabled with ZERO coefficients must be bit-for-bit identical to abv off:
// the 5eq artificial path must not perturb the state when it is switched off.
TEST(FiveEqAbv, ZeroCoeffIsNoOp) {
    const MixtureEOS m = sg_sg(1.4, 0.0, 1.6, 0.0);

    ViscousParams off;  off.mu = 0.0; off.abv_enabled = false;
    ViscousParams zero; zero.mu = 0.0; zero.abv_enabled = true;
    zero.abv_cbeta = 0.0; zero.abv_cmu = 0.0; zero.abv_ckappa = 0.0;
    zero.abv_cD = 0.0;

    std::vector<Real> snap_off, snap_zero;
    run_contact(m, 1.0, 10.0, 1.0, 1.0, 40, off,  4.0, &snap_off);
    run_contact(m, 1.0, 10.0, 1.0, 1.0, 40, zero, 4.0, &snap_zero);

    ASSERT_EQ(snap_off.size(), snap_zero.size());
    Real max_abs = 0.0;
    for (size_t i = 0; i < snap_off.size(); ++i)
        max_abs = std::max(max_abs, std::fabs(snap_off[i] - snap_zero[i]));
    EXPECT_LT(max_abs, 1e-13) << "zero-coefficient LAD perturbed the 5eq state";
}

// Shock crossing a material interface WITH LAD active: positivity/boundedness.
namespace {
struct ShockResult { Real a1min, a1max, rho_consistency; bool ok; };

ShockResult run_shock_bubble_lad(bool weno_on) {
    Grid g;
    g.nx = 200; g.ny = 4; g.nz = 4;
    g.lx = 1.0; g.ly = g.lx / g.nx * 4; g.lz = g.ly;
    g.x0 = g.y0 = g.z0 = 0.0;

    GammaLaw gl; gl.gamma = 1.4; gl.R = 1.0;
    IdealGas eos{gl};

    MixtureEOS mix = sg_sg(1.4, 0.0, 1.6, 0.0);
    const Real rho1 = 1.0, rho2 = 3.0, eps = mix.a_floor;
    const Real x_if = 0.6, delta = 3.0 * g.dx();

    State U(g.nx, g.ny, g.nz);
    FiveEqAux aux; aux.allocate(g.nx, g.ny, g.nz, U.ng());
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real x = g.xc(i);
                const Real wph = 0.5 * (1.0 + std::tanh((x_if - x) / delta));
                Real a1 = eps + (1.0 - 2.0 * eps) * wph;
                a1 = std::min(std::max(a1, eps), 1.0 - eps);
                const Real Z1 = a1 * rho1, Z2 = (1.0 - a1) * rho2;
                const Real rho = Z1 + Z2;
                const Real p = (x < 0.3) ? 10.0 : 1.0;
                const Real rhoe = mix.five_eq_rhoe_from_p(a1, Z1, Z2, p);
                U[RHO](i,j,k)  = rho;
                U[RHOU](i,j,k) = 0.0;
                U[RHOV](i,j,k) = 0.0;
                U[RHOW](i,j,k) = 0.0;
                U[RHOE](i,j,k) = rhoe;
                aux.Z1(i,j,k) = Z1; aux.Z2(i,j,k) = Z2; aux.a1(i,j,k) = a1;
            }

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Outflow;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;

    ViscousParams vp = lad_5eq(/*cD*/ 0.05, weno_on);
    vp.rho_floor = 1e-10; vp.eint_floor = 1e-10;
    const Real dxm = std::min({g.dx(), g.dy(), g.dz()});

    ShockResult sr{1e300, -1e300, 0.0, true};
    RK3 driver(g.nx, g.ny, g.nz, U.ng());
    for (int s = 0; s < 120; ++s) {
        Real dt = max_dt_hyperbolic(U, g, eos, 0.4, &aux.a1, &mix, &aux);
        const Real num = driver.last_abv_nu_max();
        if (num > 0.0) dt = std::min(dt, 0.25 * dxm * dxm / num);
        if (!(dt > 0.0) || !std::isfinite(dt)) { sr.ok = false; return sr; }
        driver.step_5eq(U, aux, g, bc, eos, vp, dt, mix);
    }

    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const Real rho = U[RHO](i,j,k);
                if (!std::isfinite(rho) || rho <= 0.0) sr.ok = false;
                const Real u = U[RHOU](i,j,k)/rho;
                const Real v = U[RHOV](i,j,k)/rho;
                const Real w = U[RHOW](i,j,k)/rho;
                const Real ke = 0.5 * rho * (u*u + v*v + w*w);
                Real p, c;
                mix.p_c_5eq(aux.a1(i,j,k), aux.Z1(i,j,k), aux.Z2(i,j,k),
                            rho, U[RHOE](i,j,k) - ke, p, c);
                if (!std::isfinite(p) || p <= 0.0) sr.ok = false;
                sr.a1min = std::min(sr.a1min, aux.a1(i,j,k));
                sr.a1max = std::max(sr.a1max, aux.a1(i,j,k));
                sr.rho_consistency = std::max(sr.rho_consistency,
                    std::fabs(rho - (aux.Z1(i,j,k) + aux.Z2(i,j,k))));
            }
    return sr;
}
}  // namespace

TEST(FiveEqAbv, ShockBubbleWithLADStaysBounded) {
    for (bool weno_on : {false, true}) {
        const ShockResult sr = run_shock_bubble_lad(weno_on);
        EXPECT_TRUE(sr.ok) << "shock+interface+LAD went non-finite/non-positive"
                           << " (weno_on=" << weno_on << ")";
        EXPECT_GE(sr.a1min, -1e-9);
        EXPECT_LE(sr.a1max, 1.0 + 1e-9);
        EXPECT_LT(sr.rho_consistency, 1e-9);
    }
}
