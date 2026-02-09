#include "test_fixtures.hpp"


TYPED_TEST(TwoGridFixture, NonUniformEdgeCorrectly) {
    grid::GenNonUniformOnRay<ViewType, Parallel> gen_line;

    geometry::Point2D<double> first_point{0.0, 0.0};

    double radius = 0.1;
    double eps = 1e-12;
    std::size_t line_size = this->grid_second.extent(0);

    kernels::fill_circle_arc_uniform(0.0, std::numbers::pi/2, radius, this->grid_first);
    geometry::Point2D<double> direction(this->grid_first(1, 0), this->grid_first(1, 1));
    geometry::Point2D<double> direction_normalized(direction.x / direction.GetL2Norm(), direction.y / direction.GetL2Norm());

    geometry::Point2D<double> start_grid_point(direction_normalized.x * 5, direction_normalized.y * 5);
    geometry::Point2D<double> end_grid_point(direction_normalized.x * 10.0, direction_normalized.y * 10.0);
    double multiplier_q = 1.2;
    gen_line(this->pthreads_pool,direction_normalized, start_grid_point, end_grid_point, multiplier_q, this->grid_second);
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