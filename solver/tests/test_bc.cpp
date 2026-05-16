#include <gtest/gtest.h>

#include "bc/BC.hpp"
#include "core/Config.hpp"
#include "core/State.hpp"

using namespace blast;

TEST(BC, PeriodicWrapsInX) {
    State U(8, 1, 1);
    for (int v = 0; v < NCONS; ++v)
        for (int i = 0; i < 8; ++i) U[v](i, 0, 0) = static_cast<Real>(i + 1);

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Periodic;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;
    apply_bcs(U, bc);

    EXPECT_DOUBLE_EQ(U[RHO](-1, 0, 0), 8.0);
    EXPECT_DOUBLE_EQ(U[RHO](-2, 0, 0), 7.0);
    EXPECT_DOUBLE_EQ(U[RHO](-3, 0, 0), 6.0);
    EXPECT_DOUBLE_EQ(U[RHO]( 8, 0, 0), 1.0);
    EXPECT_DOUBLE_EQ(U[RHO]( 9, 0, 0), 2.0);
    EXPECT_DOUBLE_EQ(U[RHO](10, 0, 0), 3.0);
}

TEST(BC, OutflowExtrapolatesInX) {
    State U(6, 1, 1);
    for (int i = 0; i < 6; ++i) U[RHO](i, 0, 0) = static_cast<Real>(i);

    BCSet bc;
    bc.xlo = bc.xhi = BCType::Outflow;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;
    apply_bcs(U, bc);

    for (int g = 1; g <= NGHOST; ++g) {
        EXPECT_DOUBLE_EQ(U[RHO](-g, 0, 0), 0.0);
        EXPECT_DOUBLE_EQ(U[RHO](5 + g, 0, 0), 5.0);
    }
}

TEST(BC, SlipWallFlipsNormalMomentumOnly) {
    State U(4, 4, 4);
    // Fill all conserved with unique distinguishable values.
    for (int v = 0; v < NCONS; ++v)
        for (int k = 0; k < 4; ++k)
            for (int j = 0; j < 4; ++j)
                for (int i = 0; i < 4; ++i)
                    U[v](i, j, k) = static_cast<Real>(v + 1 + i);

    BCSet bc;
    bc.xlo = bc.xhi = BCType::SlipWall;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.zlo = bc.zhi = BCType::Periodic;
    apply_bcs(U, bc);

    // i=-1 mirrors interior i=0, sign-flipped for RHOU only.
    EXPECT_DOUBLE_EQ(U[RHO ](-1, 1, 1),  U[RHO ](0, 1, 1));
    EXPECT_DOUBLE_EQ(U[RHOU](-1, 1, 1), -U[RHOU](0, 1, 1));
    EXPECT_DOUBLE_EQ(U[RHOV](-1, 1, 1),  U[RHOV](0, 1, 1));
    EXPECT_DOUBLE_EQ(U[RHOW](-1, 1, 1),  U[RHOW](0, 1, 1));
    EXPECT_DOUBLE_EQ(U[RHOE](-1, 1, 1),  U[RHOE](0, 1, 1));
}
