#include <gtest/gtest.h>
#include "test_fixtures.hpp"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    Kokkos::initialize(argc, argv);
    int result = RUN_ALL_TESTS();
    Kokkos::finalize();

    return result;
}