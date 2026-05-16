#include <gtest/gtest.h>

#include "core/Field3D.hpp"
#include "core/Types.hpp"

using blast::Field3D;
using blast::Real;

TEST(Field3D, ZeroInitialized) {
    Field3D f(8, 8, 8);
    for (int k = -f.ng(); k < f.nz() + f.ng(); ++k)
        for (int j = -f.ng(); j < f.ny() + f.ng(); ++j)
            for (int i = -f.ng(); i < f.nx() + f.ng(); ++i)
                ASSERT_EQ(f(i, j, k), 0.0);
}

TEST(Field3D, ReadWriteRoundTrip) {
    Field3D f(7, 5, 3);
    for (int k = 0; k < f.nz(); ++k)
        for (int j = 0; j < f.ny(); ++j)
            for (int i = 0; i < f.nx(); ++i)
                f(i, j, k) = static_cast<Real>(i + 100 * j + 10000 * k);

    for (int k = 0; k < f.nz(); ++k)
        for (int j = 0; j < f.ny(); ++j)
            for (int i = 0; i < f.nx(); ++i)
                ASSERT_EQ(f(i, j, k),
                          static_cast<Real>(i + 100 * j + 10000 * k));
}

TEST(Field3D, GhostsIndexable) {
    Field3D f(4, 4, 4);
    f(-1, 2, 3) = 11.0;
    f(4, 0, 1) = 22.0;
    EXPECT_DOUBLE_EQ(f(-1, 2, 3), 11.0);
    EXPECT_DOUBLE_EQ(f(4, 0, 1), 22.0);
}

TEST(Field3D, Alignment64B) {
    Field3D f(13, 5, 5);
    auto p = reinterpret_cast<std::uintptr_t>(f.raw());
    EXPECT_EQ(p % 64, 0U);
    EXPECT_EQ(f.ldx() % (64 / sizeof(Real)), 0);
}

TEST(Field3D, FillUniform) {
    Field3D f(6, 6, 6);
    f.fill(3.14);
    for (int k = -f.ng(); k < f.nz() + f.ng(); ++k)
        for (int j = -f.ng(); j < f.ny() + f.ng(); ++j)
            for (int i = -f.ng(); i < f.nx() + f.ng(); ++i)
                EXPECT_DOUBLE_EQ(f(i, j, k), 3.14);
}

TEST(Field3D, MoveTransfersStorage) {
    Field3D a(5, 5, 5);
    a(1, 1, 1) = 42.0;
    auto* raw_a = a.raw();

    Field3D b = std::move(a);
    EXPECT_EQ(b.raw(), raw_a);
    EXPECT_EQ(b(1, 1, 1), 42.0);
    EXPECT_EQ(a.raw(), nullptr);
}

TEST(Field3D, RejectsBadExtents) {
    Field3D f;
    EXPECT_THROW(f.resize(0, 4, 4), std::invalid_argument);
    EXPECT_THROW(f.resize(4, 4, 4, -1), std::invalid_argument);
}
