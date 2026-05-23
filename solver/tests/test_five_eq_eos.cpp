#include <gtest/gtest.h>

#include "physics/JWL.hpp"
#include "physics/MixtureEOS.hpp"
#include "physics/StiffenedGas.hpp"

#include <cmath>

using namespace blast;

// ----------------------------------------------------------------------------
// Stiffened-gas single-phase EOS.
// ----------------------------------------------------------------------------

// pinf = 0 reduces to an ideal gas: p = (gamma-1) e_int, c^2 = gamma p / rho.
TEST(StiffenedGas, IdealGasReduction) {
    StiffenedGasParams s{1.4, 0.0};
    const Real rho = 2.0, eint = 5.0;
    const Real p = sg_pressure(s, rho, eint);
    EXPECT_NEAR(p, 0.4 * eint, 1e-13);
    EXPECT_NEAR(sg_sound_speed2(s, rho, p), 1.4 * p / rho, 1e-13);
}

// p(rho,e) <-> e(rho,p) round trip, and the sound-speed formula, with pinf != 0.
TEST(StiffenedGas, RoundTripAndSoundSpeed) {
    StiffenedGasParams s{4.4, 6.0};
    const Real rho = 1000.0, p = 3.0;
    const Real eint = sg_eint_from_p(s, rho, p);
    EXPECT_NEAR(sg_pressure(s, rho, eint), p, 1e-12 * std::fabs(p) + 1e-12);
    EXPECT_NEAR(sg_sound_speed2(s, rho, p), 4.4 * (p + 6.0) / rho, 1e-12);
}

// ----------------------------------------------------------------------------
// Five-equation mixture law.
// ----------------------------------------------------------------------------

namespace {
MixtureEOS sg_sg(Real g1, Real pinf1, Real g2, Real pinf2) {
    MixtureEOS m;
    m.mode = MixMode::FiveEquation;
    m.phase[0].kind = PhaseEOS::StiffenedGas; m.phase[0].sg = {g1, pinf1};
    m.phase[1].kind = PhaseEOS::StiffenedGas; m.phase[1].sg = {g2, pinf2};
    return m;
}
}  // namespace

// When both phases are the same material, the mixture pressure equals the
// single-phase value for ANY volume fraction.
TEST(FiveEqEOS, SinglePhaseLimit) {
    const MixtureEOS m = sg_sg(1.4, 0.0, 1.4, 0.0);
    const Real rhoe = 7.0;
    for (Real a1 : {0.1, 0.5, 0.9}) {
        const Real rho1 = 2.0, rho2 = 2.0;
        const Real Z1 = a1 * rho1, Z2 = (1.0 - a1) * rho2, rho = Z1 + Z2;
        Real p, c;
        m.p_c_5eq(a1, Z1, Z2, rho, rhoe, p, c);
        EXPECT_NEAR(p, 0.4 * rhoe, 1e-12);            // ideal gas, gamma=1.4
        EXPECT_NEAR(c * c, 1.4 * p / rho, 1e-10);     // mass-weighted == single
    }
}

// Volume-fraction-averaged (Wood) pressure rule vs a direct hand computation.
TEST(FiveEqEOS, WoodRuleMatchesHandCalc) {
    const Real g1 = 1.4, pinf1 = 0.0, g2 = 3.0, pinf2 = 4.0;
    const MixtureEOS m = sg_sg(g1, pinf1, g2, pinf2);
    const Real a1 = 0.3, rho1 = 1.0, rho2 = 50.0;
    const Real Z1 = a1 * rho1, Z2 = (1.0 - a1) * rho2, rho = Z1 + Z2;
    const Real rhoe = 12.0;

    const Real S  = a1 / (g1 - 1.0) + (1.0 - a1) / (g2 - 1.0);
    const Real Pi = a1 * g1 * pinf1 / (g1 - 1.0)
                  + (1.0 - a1) * g2 * pinf2 / (g2 - 1.0);
    const Real p_hand = (rhoe - Pi) / S;

    Real p, c;
    m.p_c_5eq(a1, Z1, Z2, rho, rhoe, p, c);
    EXPECT_NEAR(p, p_hand, 1e-10 * std::fabs(p_hand));
    EXPECT_GT(c, 0.0);
}

// IC self-consistency: rhoe from a target p reproduces that p exactly.
TEST(FiveEqEOS, RhoeFromPressureRoundTrip) {
    const MixtureEOS m = sg_sg(1.4, 0.0, 4.4, 6.0);
    const Real a1 = 0.6, rho1 = 1.2, rho2 = 800.0;
    const Real Z1 = a1 * rho1, Z2 = (1.0 - a1) * rho2, rho = Z1 + Z2;
    const Real p_target = 2.5;
    const Real rhoe = m.five_eq_rhoe_from_p(a1, Z1, Z2, p_target);
    Real p, c;
    m.p_c_5eq(a1, Z1, Z2, rho, rhoe, p, c);
    EXPECT_NEAR(p, p_target, 1e-11 * p_target);
}

// A JWL phase is usable in the mixture: with the JWL reference curve A=B=0 it is
// an ideal gas of gamma = 1+omega, so an SG(gamma=1+omega)/JWL pair must agree.
TEST(FiveEqEOS, JWLPhaseReducesToIdealMix) {
    const Real omega = 0.4;
    MixtureEOS mj;
    mj.mode = MixMode::FiveEquation;
    mj.phase[0].kind = PhaseEOS::StiffenedGas; mj.phase[0].sg = {1.4, 0.0};
    mj.phase[1].kind = PhaseEOS::JWLPhase;
    mj.phase[1].jwl.A = 0.0; mj.phase[1].jwl.B = 0.0;
    mj.phase[1].jwl.R1 = 4.0; mj.phase[1].jwl.R2 = 1.0;   // nonzero (avoid 0/0)
    mj.phase[1].jwl.omega = omega; mj.phase[1].jwl.rho0 = 1.0;

    const MixtureEOS ms = sg_sg(1.4, 0.0, 1.0 + omega, 0.0);

    const Real a1 = 0.45, rho1 = 1.0, rho2 = 3.0;
    const Real Z1 = a1 * rho1, Z2 = (1.0 - a1) * rho2, rho = Z1 + Z2;
    const Real rhoe = 9.0;
    Real pj, cj, ps, cs;
    mj.p_c_5eq(a1, Z1, Z2, rho, rhoe, pj, cj);
    ms.p_c_5eq(a1, Z1, Z2, rho, rhoe, ps, cs);
    EXPECT_NEAR(pj, ps, 1e-10 * std::fabs(ps));
    EXPECT_NEAR(cj, cs, 1e-10 * cs);
}
