#include "test_fixtures.hpp"

TEST(InterceptionTest, Vertical) {
    using p_type = geometry::Point2D<double>;
    p_type true_interception(3, 6);
    auto interception = geometry::interception_lines(
                                            p_type{0.0, 0.0}, p_type{1.0, 2.0},
                                            p_type{3.0, 0.0}, p_type{3.0, 2.0}
                                            );
    EXPECT_TRUE(
        std::abs(true_interception.x - interception.x) < std::numeric_limits<double>::epsilon() &&
        std::abs(true_interception.y - interception.y) < std::numeric_limits<double>::epsilon()
        )   << "interception_lines gave wrong result:\n"
            << "Expected: " << true_interception.x << " " << true_interception.y << "\n"
            << "Got     : " << interception.x << " " << interception.y << "\n";
}

TEST(InterceptionTest, SamePoint) {
    using p_type = geometry::Point2D<double>;
    p_type true_interception(3, 6);
    auto interception = geometry::interception_lines(
                                            p_type{0.0, 0.0}, p_type{3.0, 6.0},
                                            p_type{3.0, 0.0}, p_type{3.0, 6.0}
                                            );
    EXPECT_TRUE(
        std::abs(true_interception.x - interception.x) < std::numeric_limits<double>::epsilon() &&
        std::abs(true_interception.y - interception.y) < std::numeric_limits<double>::epsilon()
        )   << "interception_lines gave wrong result:\n"
            << "Expected: " << true_interception.x << " " << true_interception.y << "\n"
            << "Got     : " << interception.x << " " << interception.y << "\n";
}