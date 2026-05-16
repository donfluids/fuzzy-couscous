#include <gtest/gtest.h>

#include "core/Config.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "ic/Canonical.hpp"
#include "io/Restart.hpp"
#include "physics/EOS.hpp"

#include <cstdio>
#include <filesystem>

using namespace blast;

TEST(Restart, RoundTripRecoversInteriorAndHeader) {
    Grid g; g.nx = 12; g.ny = 8; g.nz = 6;
    g.lx = 1.0; g.ly = 1.0; g.lz = 1.0;
    g.x0 = 0.0; g.y0 = 0.0; g.z0 = 0.0;
    IdealGas eos{GammaLaw{}};

    State A(g.nx, g.ny, g.nz);
    ic_sphere_blast_3d(A, g, eos, 10.0, 50.0, 1.0, 1.0, 0.2, 0.05, 0.1);

    const std::string path = (std::filesystem::temp_directory_path()
                              / "blast_test_restart.ckpt.h5").string();
    write_checkpoint(path, A, g, /*t=*/0.0125, /*step=*/77);

    State B(g.nx, g.ny, g.nz);
    auto header = read_checkpoint(path, B, g);
    EXPECT_DOUBLE_EQ(header.time, 0.0125);
    EXPECT_EQ(header.step, 77);
    EXPECT_EQ(header.nx, g.nx);
    EXPECT_EQ(header.ny, g.ny);
    EXPECT_EQ(header.nz, g.nz);

    for (int v = 0; v < NCONS; ++v)
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j)
                for (int i = 0; i < g.nx; ++i)
                    ASSERT_DOUBLE_EQ(A[v](i, j, k), B[v](i, j, k))
                        << "mismatch at (" << v << "," << i << "," << j << "," << k << ")";

    std::remove(path.c_str());
}

TEST(Restart, RejectsShapeMismatch) {
    Grid g_write; g_write.nx = 8; g_write.ny = 8; g_write.nz = 8;
    g_write.lx = 1.0; g_write.ly = 1.0; g_write.lz = 1.0;
    IdealGas eos{GammaLaw{}};
    State A(g_write.nx, g_write.ny, g_write.nz);
    ic_sphere_blast_3d(A, g_write, eos, 2.0, 5.0, 1.0, 1.0, 0.2, 0.0, 0.0);

    const std::string path = (std::filesystem::temp_directory_path()
                              / "blast_test_restart_bad.ckpt.h5").string();
    write_checkpoint(path, A, g_write, 0.0, 0);

    Grid g_read = g_write; g_read.nx = 16;       // wrong shape
    State B(g_read.nx, g_read.ny, g_read.nz);
    EXPECT_THROW(read_checkpoint(path, B, g_read), std::runtime_error);

    std::remove(path.c_str());
}
