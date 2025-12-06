#include "test_utils.hpp"

TEST(InterceptionTest, Vertical) {
    using p_type = grid::Point2D<double>;
    p_type true_interception(3, 6);
    auto interception = grid::interception_lines(
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
    using p_type = grid::Point2D<double>;
    p_type true_interception(3, 6);
    auto interception = grid::interception_lines(
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

TYPED_TEST(SingleGridFixture, FirstLastCircleCorrectly) {
    grid::GenUniformOnCircle<ViewType> generate;
    double radius = 0.1;
    generate(0.0, std::numbers::pi/2, radius, this->grid);
    std::size_t size_grid = this->grid.extent(0);
    EXPECT_TRUE(
        std::abs(this->grid(size_grid - 1, 0) - 0.0) < std::numeric_limits<double>::epsilon() &&
        std::abs(this->grid(size_grid - 1, 1) - radius) < std::numeric_limits<double>::epsilon() &&

        std::abs(this->grid(0, 0) - radius) < std::numeric_limits<double>::epsilon() &&
        std::abs(this->grid(0, 1) - 0.0) < std::numeric_limits<double>::epsilon()
        )   << "GenUniformOnCircle gave wrong result:\n"
            << "Expected: " << 0.0 << " " << radius << " " << radius << " " << 0.0 << "\n"
            << "Got     : " << this->grid(size_grid - 1, 0) << " " << this->grid(size_grid - 1, 1) << " " << this->grid(0, 0) - radius << " " << this->grid(0, 1) - 0.0 << "\n";
}

TYPED_TEST(SingleGridFixture, StepCorrectly) {
    grid::GenUniformOnCircle<ViewType> generate;
    double radius = 0.1;
    generate(0.0, std::numbers::pi/2, radius, this->grid);
    std::size_t size_grid = this->grid.extent(0);
    double x1_x0 = std::pow(this->grid(1, 0) - this->grid(0, 0), 2);
    double y1_y0 = std::pow(this->grid(1, 1) - this->grid(0, 1), 2);
    double step_size = std::sqrt(x1_x0 + y1_y0);
    for (std::size_t i = 2; i < size_grid; i++) {
        double xi1_xi = std::pow(this->grid(i, 0) - this->grid(i-1, 0), 2);
        double yi1_yi = std::pow(this->grid(i, 1) - this->grid(i-1, 1), 2);
        double i_step_size = std::sqrt(xi1_xi + yi1_yi);
        EXPECT_TRUE(
        std::abs(step_size - i_step_size) < std::numeric_limits<double>::epsilon()
        )   << "GenUniformOnCircle gave wrong result:\n"
            << "Expected: " << step_size  << "\n"
            << "Got     : " << i_step_size << "\n";
    }

}

TYPED_TEST(TwoGridFixture, NonUniformEdgeCorrectly) {
    grid::GenNonUniformOnLine<ViewType, Parallel> gen_line;
    grid::GenUniformOnCircle<ViewType> gen_circle;

    grid::Point2D<double> first_point{0.0, 0.0};

    double radius = 0.1;
    double eps = 1e-12;
    std::size_t line_size = this->grid_second.extent(0);

    gen_circle(0.0, std::numbers::pi/2, radius, this->grid_first);
    grid::Point2D<double> direction(this->grid_first(1, 0), this->grid_first(1, 1));
    grid::Point2D<double> direction_normalized(direction.x / direction.GetL2Norm(), direction.y / direction.GetL2Norm());

    grid::Point2D<double> start_grid_point(direction_normalized.x * 5, direction_normalized.y * 5);
    grid::Point2D<double> end_grid_point(direction_normalized.x * 10.0, direction_normalized.y * 10.0);
    double multiplier_q = 1.2;
    gen_line(direction_normalized, start_grid_point, end_grid_point, multiplier_q, this->grid_second);
    EXPECT_TRUE(
        std::abs(this->grid_second(0, 0) - start_grid_point.x) < eps &&
        std::abs(this->grid_second(0, 1) - start_grid_point.y) < eps &&

        std::abs(this->grid_second(line_size - 1, 0) - end_grid_point.x) < eps &&
        std::abs(this->grid_second(line_size - 1, 1) - end_grid_point.y) < eps
        )   << "GenNonUniformOnLine gave wrong result:\n"
            << "Expected: " << start_grid_point.x  << " " << start_grid_point.y << " " << end_grid_point.x << " " << end_grid_point.y << "\n"
            << "Got     : " << this->grid_second(0, 0)  << " " << this->grid_second(0, 1) << " " << this->grid_second(line_size - 1, 0) << " " << this->grid_second(line_size - 1, 1) << "\n";

    for (std::size_t i = 0; i < line_size - 1; i++) {
        EXPECT_TRUE(this->grid_second(i, 0) < end_grid_point.x && this->grid_second(i, 1) < end_grid_point.y);
    }


}