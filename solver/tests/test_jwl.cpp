#include <gtest/gtest.h>

#include "bc/BC.hpp"
#include "core/Config.hpp"
#include "core/Field3D.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "numerics/RHS.hpp"
#include "numerics/RK3.hpp"
#include "physics/EOS.hpp"
#include "physics/JWL.hpp"
#include "physics/MixtureEOS.hpp"
#include "physics/Multifluid.hpp"
#include "physics/ViscousFlux.hpp"

#include <cmath>

using namespace blast;

namespace {
// Standard LLNL/Dobratz TNT JWL parameters (SI). Self-consistent with
// p_CJ = 21 GPa, D = 6930 m/s, rho0 = 1630 kg/m^3 (checked below).
JWLParams tnt_jwl() {
    JWLParams j;
    j.A = 3.712e11; j.B = 3.231e9;
    j.R1 = 4.15; j.R2 = 0.95; j.omega = 0.30;
    j.rho0 = 1630.0; j.E0 = 7.0e9;
    return j;
}

// Nondimensional TNT MultifluidParams (rho_ref=1630, p_ref=21e9) for the IC
// and integration tests: products O(1), air ~ 1e-3.
MultifluidParams tnt_nondim_params() {
    const Real rr = 1630.0, pr = 21.0e9;
    MultifluidParams mp;
    mp.enabled = true; mp.gamma_air = 1.4; mp.R = 1.0;
    mp.jwl_mode = true; mp.phi_switch = 0.5;
    mp.jwl.A = 3.712e11 / pr; mp.jwl.B = 3.231e9 / pr;
    mp.jwl.R1 = 4.15; mp.jwl.R2 = 0.95; mp.jwl.omega = 0.30;
    mp.jwl.rho0 = 1630.0 / rr; mp.jwl.E0 = 7.0e9 / pr;
    mp.rho_cj = 2228.0 / rr; mp.p_cj = 21.0e9 / pr;
    mp.rho_a = 1.2 / rr; mp.p_a = 1.013e5 / pr;
    mp.r0 = 0.2; mp.tanh_thickness = 0.01; mp.Y42_amp = 0.0;
    return mp;
}
}  // namespace

// ----------------------------------------------------------------------------
// Phase A gate: enabling a TwoGamma MixtureEOS must be a bit-for-bit no-op vs
// the original gloc = 1 + 1/G arithmetic (mix == nullptr).
// ----------------------------------------------------------------------------
TEST(MixtureEOS, TwoGammaMatchesLegacyGfnPathBitExact) {
    GammaLaw gl; gl.gamma = 1.4; gl.R = 1.0;
    IdealGas eos{gl};

    Grid g; g.nx = g.ny = g.nz = 32;
    g.lx = g.ly = g.lz = 1.0; g.x0 = g.y0 = g.z0 = -0.5;

    MultifluidParams mp;
    mp.enabled = true; mp.gamma_air = 1.4; mp.gamma_p = 1.25; mp.R = 1.0;
    mp.rho_p = 10.0; mp.T_p = 100.0; mp.rho_a = 1.0; mp.T_a = 1.0;
    mp.r0 = 0.1; mp.tanh_thickness = 0.012; mp.Y42_amp = 0.2;

    State U(g.nx, g.ny, g.nz);
    Field3D G(g.nx, g.ny, g.nz);
    mf_init_blast(U, G, g, mp);

    BCSet bc;
    bc.xlo = bc.xhi = bc.ylo = bc.yhi = bc.zlo = bc.zhi = BCType::SlipWall;
    apply_bcs(U, bc);
    mf_fill_G_bcs(G, bc);

    MixtureEOS mix;            // mode defaults to TwoGamma
    mix.mode = MixMode::TwoGamma;
    mix.gamma_air = 1.4;

    State R_legacy(g.nx, g.ny, g.nz), R_mix(g.nx, g.ny, g.nz);
    compute_rhs_inviscid(U, g, eos, R_legacy, &G, nullptr);   // gloc = 1 + 1/G
    compute_rhs_inviscid(U, g, eos, R_mix,    &G, &mix);       // MixtureEOS path

    // The flux divergence (the actual RK update increment) is bit-for-bit
    // identical: the TwoGamma branch of MixtureEOS reproduces gloc = 1 + 1/G
    // exactly.
    double maxabs = 0.0;
    for (int v = 0; v < NCONS; ++v)
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j)
                for (int i = 0; i < g.nx; ++i)
                    maxabs = std::max(maxabs,
                        std::fabs(R_legacy[v](i,j,k) - R_mix[v](i,j,k)));
    EXPECT_EQ(maxabs, 0.0);

    // dt is a max-reduction over the per-cell sound speed. Routing the same
    // arithmetic through the inlined p_c() member lets -ffast-math reassociate
    // a single op differently than the hand-inlined loop, so dt can land 1 ULP
    // apart even though the flux divergence is bit-identical. Require round-off
    // agreement (observed ~5e-19 relative).
    const Real dt_legacy = max_dt_hyperbolic(U, g, eos, 0.3, &G, nullptr);
    const Real dt_mix    = max_dt_hyperbolic(U, g, eos, 0.3, &G, &mix);
    EXPECT_NEAR(dt_legacy, dt_mix, 1e-14 * dt_legacy);
}

// ----------------------------------------------------------------------------
// JWL EOS self-consistency.
// ----------------------------------------------------------------------------

// p(rho,e) and the e(rho,p) inverse round-trip; sound speed is real.
TEST(JWL, PressureEnergyRoundTrip) {
    const JWLParams j = tnt_jwl();
    const Real rho = 2000.0, p = 1.5e10;
    const Real eint = jwl_eint_from_p(j, rho, p);
    EXPECT_NEAR(jwl_pressure(j, rho, eint), p, 1e-6 * p);
    EXPECT_GT(jwl_sound_speed2(j, rho, p), 0.0);
}

// With no reference curve (A=B=0) JWL reduces to an ideal gas of gamma = 1+omega:
// p = omega*e_int, c^2 = (1+omega) p / rho. Validates the c^2 derivation.
TEST(JWL, IdealGasReduction) {
    JWLParams j; j.A = 0.0; j.B = 0.0; j.R1 = 4.0; j.R2 = 1.0;
    j.omega = 0.4; j.rho0 = 1.0;
    const Real rho = 2.0, eint = 5.0;
    const Real p = jwl_pressure(j, rho, eint);
    EXPECT_NEAR(p, 0.4 * eint, 1e-12);
    EXPECT_NEAR(jwl_sound_speed2(j, rho, p), 1.4 * p / rho, 1e-12);
}

// Non-tautological consistency: at TNT's CJ point the JWL sound speed must equal
// the value implied by the CJ condition D - u_CJ = c_CJ. With the Rankine-
// Hugoniot mass/momentum relations across the front (p0=0, u0=0):
//   rho0/rho_CJ = 1 - p_CJ/(rho0 D^2),   u_CJ = D(1 - rho0/rho_CJ),
//   so c_CJ = D - u_CJ = D * (rho0/rho_CJ).
// Tests both the parameter set AND the c^2 implementation simultaneously.
TEST(JWL, TNT_CJ_SoundSpeedConsistency) {
    const JWLParams j = tnt_jwl();
    const Real D = 6930.0, p_cj = 21.0e9;
    const Real ratio  = 1.0 - p_cj / (j.rho0 * D * D);   // rho0 / rho_CJ
    const Real rho_cj = j.rho0 / ratio;
    const Real c_jwl  = std::sqrt(jwl_sound_speed2(j, rho_cj, p_cj));
    const Real c_cj   = D * ratio;                        // D - u_CJ
    EXPECT_NEAR(c_jwl, c_cj, 1e-2 * c_cj);               // within 1%
}

// ----------------------------------------------------------------------------
// JWL CJ initial condition: the products interior carries the tabulated CJ
// state, and recomputing pressure from the JWL EOS returns p_cj. phi bounded.
// ----------------------------------------------------------------------------
TEST(JWL, CJ_InitialConditionSelfConsistent) {
    Grid g; g.nx = g.ny = g.nz = 32;
    g.lx = g.ly = g.lz = 1.0; g.x0 = g.y0 = g.z0 = -0.5;

    const MultifluidParams mp = tnt_nondim_params();
    State U(g.nx, g.ny, g.nz);
    Field3D G(g.nx, g.ny, g.nz);
    mf_init_blast(U, G, g, mp);

    // Center cell is well inside r0=0.2 -> pure products (phi=1).
    const int ic = g.nx / 2, jc = g.ny / 2, kc = g.nz / 2;
    EXPECT_NEAR(G(ic, jc, kc), 1.0, 1e-6);
    EXPECT_NEAR(U[RHO](ic, jc, kc), mp.rho_cj, 1e-6 * mp.rho_cj);
    // Velocity is zero -> e_int = rhoE. Recompute pressure from JWL.
    const Real eint = U[RHOE](ic, jc, kc);
    const Real p = jwl_pressure(mp.jwl, U[RHO](ic, jc, kc), eint);
    EXPECT_NEAR(p, mp.p_cj, 1e-6 * mp.p_cj);

    // phi is a mass fraction and must stay in [0,1] everywhere.
    Real gmin = 1e300, gmax = -1e300;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                gmin = std::min(gmin, G(i,j,k));
                gmax = std::max(gmax, G(i,j,k));
            }
    EXPECT_GE(gmin, -1e-12);
    EXPECT_LE(gmax, 1.0 + 1e-12);
}

// A few steps of the full JWL multifluid path stay bounded and finite: phi in
// [0,1], rho > 0, pressure positive, no NaN at the 1000:1 contact.
TEST(JWL, ShortRunStaysBoundedAndFinite) {
    GammaLaw gl; gl.gamma = 1.4; gl.R = 1.0;
    IdealGas eos{gl};

    Grid g; g.nx = g.ny = g.nz = 32;
    g.lx = g.ly = g.lz = 1.0; g.x0 = g.y0 = g.z0 = -0.5;

    const MultifluidParams mp = tnt_nondim_params();
    State U(g.nx, g.ny, g.nz);
    Field3D G(g.nx, g.ny, g.nz);
    G.fill(0.0);
    mf_init_blast(U, G, g, mp);

    BCSet bc;
    bc.xlo = bc.xhi = bc.ylo = bc.yhi = bc.zlo = bc.zhi = BCType::Outflow;
    mf_fill_G_bcs(G, bc);
    apply_bcs(U, bc);

    MixtureEOS mix = mp.mixture();
    ViscousParams vp; vp.mu = 0.0;
    vp.rho_floor  = 1e-3 * mp.rho_a;
    vp.eint_floor = 1e-3 * (mp.p_a / (mp.gamma_air - 1.0));

    RK3 driver(g.nx, g.ny, g.nz, U.ng());
    for (int s = 0; s < 10; ++s) {
        const Real dt = max_dt_hyperbolic(U, g, eos, 0.3, &G, &mix);
        ASSERT_GT(dt, 0.0);
        ASSERT_TRUE(std::isfinite(dt));
        driver.step(U, g, bc, eos, vp, dt, &G, &mix);
        mf_advect_G(G, U, g, bc, dt);
    }

    Real pmin, pmax;
    mf_pressure_minmax(U, G, pmin, pmax, &mix);
    EXPECT_GT(pmin, 0.0);
    EXPECT_TRUE(std::isfinite(pmax));
    const GStats gs = mf_g_stats(G);
    EXPECT_GE(gs.gmin, -1e-9);
    EXPECT_LE(gs.gmax, 1.0 + 1e-9);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
                ASSERT_TRUE(std::isfinite(U[RHO](i,j,k)) && U[RHO](i,j,k) > 0.0);
}
