#include "test_fixtures.hpp"

// TEST(DISABLED_PartitionalTest, FixSizeCorrectWithOneSlave) {
//     std::size_t N = 21;
//     auto alloc = Kokkos::view_alloc(Kokkos::WithoutInitializing, "v");
//     auto mesh = ViewType(alloc, N);
//     for (std::size_t i = 0; i < N; i++) {
//         mesh(i, 0) = i;
//         mesh(i, 1) = i;
//     }
//     auto sector = mesh::prepare_sector(2, 1, 3, mesh);
//     std::size_t sum = 0;
//     for (std::size_t i = 0; i < sector.extent(0); i++)
//         sum += sector(i, 0);
//     EXPECT_EQ(sum, 144)
//             << "prepare_sector gave wrong result:\n"
//             << "Expected: " << 144  << "\n"
//             << "Got     : " << sum << "\n";
// }
//
// TEST(DISABLED_PartitionalTest, FixSizeCorrectWithTwoSlave) {
//     std::size_t N = 21;
//     auto alloc = Kokkos::view_alloc(Kokkos::WithoutInitializing, "v");
//     auto mesh = ViewType(alloc, N);
//     for (std::size_t i = 0; i < N; i++) {
//         mesh(i, 0) = i;
//         mesh(i, 1) = i;
//     }
//     auto sector = mesh::prepare_sector(1, 2, 3, mesh);
//     std::size_t sum = 0;
//     for (std::size_t i = 0; i < sector.extent(0); i++)
//         sum += sector(i, 0);
//     EXPECT_EQ(sum, 174)
//             << "prepare_sector gave wrong result:\n"
//             << "Expected: " << 174  << "\n"
//             << "Got     : " << sum << "\n";
// }