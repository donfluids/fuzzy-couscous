#include <gtest/gtest.h>

#include "core/Config.hpp"

using blast::BCType;
using blast::ICType;

TEST(Config, LoadsExampleFixture) {
    auto c = blast::load_config("fixtures/example.toml");

    EXPECT_EQ(c.run_name, "chamber_smoke");
    EXPECT_EQ(c.grid.nx, 64);
    EXPECT_EQ(c.grid.ny, 64);
    EXPECT_EQ(c.grid.nz, 64);
    EXPECT_DOUBLE_EQ(c.grid.lx, 0.5);
    EXPECT_EQ(c.bc.xlo, BCType::SlipWall);
    EXPECT_EQ(c.bc.xhi, BCType::SlipWall);
    EXPECT_EQ(c.ic.type, ICType::TopHatSphere);
    EXPECT_DOUBLE_EQ(c.physics.eos.gamma, 1.4);
    EXPECT_TRUE(c.afp.enabled);
}

TEST(Config, RejectsUnknownBC) {
    EXPECT_THROW(blast::load_config("fixtures/nonexistent.toml"),
                 std::runtime_error);
}
