#pragma once
#include <cmath>
#include "../math/math_helper.hpp"
#include "../custom_concepts.hpp"
#include "core/geometry/geometry.hpp"

namespace kernels {

    /**
     * Функция для заполнения переданного интервала сетки равномерной сеткой на окружности
     * @tparam ContainerT
     * @tparam ScalarT
     * @param start_arc_angle_rad Угол в радианах начала дуги
     * @param end_arc_angle_rad Угол в радианах конца дуги
     * @param radius
     * @param circle_storage
     */
    template <kokkos_view_2d_like ContainerT, typename ScalarT>
    void fill_circle_arc_uniform(
                            ScalarT start_arc_angle_rad,
                            ScalarT end_arc_angle_rad,
                            ScalarT radius,
                            ContainerT circle_storage
                        ) noexcept {
        const std::size_t grid_size = circle_storage.extent(0);
        const ScalarT step_on_circle = (end_arc_angle_rad - start_arc_angle_rad) / (grid_size - 1);
        for (std::size_t i = 0; i < grid_size; i++) {
            circle_storage(i, 0) = radius * std::cos(start_arc_angle_rad + i * step_on_circle);
            circle_storage(i, 1) = radius * std::sin(start_arc_angle_rad + i * step_on_circle);
        }
    }

    /**
     * Функция для заполнения переданного интервала сетки неравномерной сеткой на основе геометрической прогрессии
     * (x, y) = (x_0, y_0) + (V_x, V_y) * (b-a) * t уравнение прямой, где t = (1 - q^i) / 1 - q^N
     * @tparam ContainerT
     * @tparam ScalarT
     * @param normalized_direction_ray Вектор направления луча
     * @param start_point_grid Точка на луче, с которой начинается заполнение сетки (должна лежать на направляющем векторе)
     * @param end_point_grid Точка на луче, на которой заканчивается заполнение сетки (должна лежать на направляющем векторе)
     * @param multiplier_q Основание геометрической прогрессии для роста сетки
     * @param denominator (1 - q^N)
     * @param idx_from_start Индекс от начала (tid * standard_segment_size), требуется в показателе степени (1 - q^i)
     * @param ray_segment_storage
     */
    template <kokkos_view_2d_like ContainerT, typename ScalarT>
    void fill_ray_segment_nonuniform(
                const geometry::Point2D<ScalarT> &normalized_direction_ray,
                const geometry::Point2D<ScalarT> &start_point_grid,
                const geometry::Point2D<ScalarT> &end_point_grid,
                ScalarT multiplier_q,
                ScalarT denominator,
                std::size_t idx_from_start,
                ContainerT ray_segment_storage
                ) noexcept {
        auto interval = end_point_grid - start_point_grid;
        ScalarT length_interval = interval.GetL2Norm();
        const std::size_t subgrid_size = ray_segment_storage.extent(0);

        for (std::size_t i = 0; i < subgrid_size; i++) {
            //(x, y) = (x_0, y_0) + (V_x, V_y) * (b-a) * t уравнение прямой
            //t = (1 - q^i) / 1 - q^N
            ray_segment_storage(i, 0) = //x
                start_point_grid.x + normalized_direction_ray.x * length_interval *
                        (ScalarT(1) - math_helper::fast_pow(multiplier_q, idx_from_start + i)) / denominator;
            ray_segment_storage(i, 1) = //y
                start_point_grid.y + normalized_direction_ray.y * length_interval *
                        (ScalarT(1) - math_helper::fast_pow(multiplier_q, idx_from_start + i)) / denominator;
        }
    }
}
