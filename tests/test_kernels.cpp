#include "test_fixtures.hpp"

TYPED_TEST(SingleGridFixture, FirstLastCircleCorrectly) {
    double radius = 0.1;
    kernels::fill_circle_arc_uniform(0.0, std::numbers::pi/2, radius, this->grid);
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

TYPED_TEST(SingleGridFixture, StepCircleCorrectly) {
    double radius = 0.1;
    kernels::fill_circle_arc_uniform(0.0, std::numbers::pi/2, radius, this->grid);
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

TYPED_TEST(SingleGridFixture, StepNonUniformCorrectly) {
    using p_type = geometry::Point2D<double>;
    p_type direction{0.5, 0.5};
    direction.Normalize();
    double multiplier = 1.2;
    std::size_t N = this->grid.extent(0);
    double denominator = 1 - math_helper::fast_pow(multiplier, N - 1);
    kernels::fill_ray_segment_nonuniform(
                    direction,
                    p_type{direction.x * 3.4, direction.y * 3.4},
                    p_type{direction.x * 9.6, direction.y * 9.6},
                    multiplier,
                    denominator,
                    0,
                    this->grid
                    );
    double x1_x0 = std::pow(this->grid(1, 0) - this->grid(0, 0), 2);
    double y1_y0 = std::pow(this->grid(1, 1) - this->grid(0, 1), 2);
    double eps = 1e-12;
    double step_size = std::sqrt(x1_x0 + y1_y0);
    for (std::size_t i = 2; i < N; i++) {
        double xi1_xi = std::pow(this->grid(i, 0) - this->grid(i-1, 0), 2);
        double yi1_yi = std::pow(this->grid(i, 1) - this->grid(i-1, 1), 2);
        double i_step_size = std::sqrt(xi1_xi + yi1_yi);
        EXPECT_TRUE(
        std::abs(i_step_size - step_size * multiplier) < eps
        )   << "fill_ray_segment_nonuniform gave wrong result:\n"
            << "Expected: " << step_size * multiplier  << "\n"
            << "Got     : " << i_step_size << "\n";
        step_size = i_step_size;
    }
}