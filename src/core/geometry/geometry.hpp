#pragma once
#include <cmath>

namespace geometry {
    template<typename ScalarT>
    struct Point2D {
        using value_type = ScalarT;
        ScalarT x, y;

        constexpr Point2D() = default;
        constexpr Point2D(ScalarT first, ScalarT second) : x(first), y(second) {}

        [[nodiscard]] constexpr Point2D operator-(const Point2D &other) const noexcept{
            Point2D result;
            result.x = this->x - other.x;
            result.y = this->y - other.y;
            return result;
        }
        [[nodiscard]] ScalarT GetL2Norm() const noexcept { return std::hypot(x, y); }

        void Normalize() noexcept {
            ScalarT norm = this->GetL2Norm();
            this->x /= norm;
            this->y /= norm;
        }
    };

    /**
     * Функция для поиска пересечения двух прямых, заданных через 2 точки.
     * Прямые не должны быть параллельными/околопараллельными
     * @tparam ScalarT
     * @param point_1
     * @param point_2
     * @param point_3
     * @param point_4
     * @return Point2D
     */
    template<typename ScalarT>
    [[nodiscard]] Point2D<ScalarT> interception_lines(
                                const Point2D<ScalarT>& point_1,
                                const Point2D<ScalarT>& point_2,
                                const Point2D<ScalarT>& point_3,
                                const Point2D<ScalarT>& point_4
                                ) noexcept {
        ScalarT x1_x2 = point_1.x - point_2.x;
        ScalarT x3_x4 = point_3.x - point_4.x;

        ScalarT y1_y2 = point_1.y - point_2.y;
        ScalarT y3_y4 = point_3.y - point_4.y;

        ScalarT denominator = x1_x2 * y3_y4 - y1_y2 * x3_x4;

        ScalarT x1y2_y1x2 = point_1.x * point_2.y - point_1.y * point_2.x;
        ScalarT x3y4_y3x4 = point_3.x * point_4.y - point_3.y * point_4.x;

        Point2D<ScalarT> result;
        result.x = (x1y2_y1x2 * x3_x4 - x1_x2 * x3y4_y3x4) / denominator;
        result.y = (x1y2_y1x2 * y3_y4 - y1_y2 * x3y4_y3x4) / denominator;

        return result;
    }
}