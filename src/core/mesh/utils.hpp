#pragma once
#include <iostream>
#include <Kokkos_Core.hpp>
#include <numbers>
#include "../custom_concepts.hpp"

namespace grid {
    template<typename ScalarT>
    struct Point2D {
        using value_type = ScalarT;
        ScalarT x, y;

        Point2D() = default;
        Point2D(ScalarT first, ScalarT second) : x(first), y(second) {}
        [[nodiscard]] Point2D operator-(const Point2D &other) const {
            Point2D result;
            result.x = this->x - other.x;
            result.y = this->y - other.y;
            return result;
        }
        [[nodiscard]] ScalarT GetL2Norm() const{ return std::sqrt(this->x * this->x + this->y * this->y); }

    };
    template<typename ScalarT>
    Point2D<ScalarT> interception_lines(
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

    template<typename ScalarT>
    constexpr ScalarT fast_pow(ScalarT base, std::size_t exp) noexcept {
        if (exp == 0)
            return ScalarT(1);

        ScalarT result = ScalarT(1);
        while (exp > 0) {
            if (exp & 1)
                result *= base;

            base *= base;
            exp >>= 1;
        }

        return result;
    }

    template <kokkos_view_2d_like ContainerT>
    struct ThreadArgs {
        using ScalarT = ContainerT::value_type;

        ContainerT grid_subrange_;
        Point2D<ScalarT> normalized_direction_;
        Point2D<ScalarT> start_point_grid_, end_point_grid_;
        ScalarT multiplier_q_;
        ScalarT denominator_;
        std::size_t standard_subgrid_size_;
        std::size_t thread_id_;

        ThreadArgs(
                ContainerT grid_subrange,
                Point2D<ScalarT> normalized_direction,
                Point2D<ScalarT> start_point_grid,
                Point2D<ScalarT> end_point_grid,
                ScalarT multiplier_q,
                ScalarT denominator,
                std::size_t standard_subgrid_size,
                std::size_t tid
            ) noexcept :
        grid_subrange_(grid_subrange),
        normalized_direction_(normalized_direction),
        start_point_grid_(start_point_grid),
        end_point_grid_(end_point_grid),
        multiplier_q_(multiplier_q),
        denominator_(denominator),
        standard_subgrid_size_(standard_subgrid_size),
        thread_id_(tid) {}

    };

    template <kokkos_view_2d_like ContainerT>
    void* worker(void* args) {
        using ScalarT = ContainerT::value_type;
        auto args_ptr = static_cast<ThreadArgs<ContainerT>*>(args);

        auto subgrid_storage = args_ptr->grid_subrange_;

        auto normalized_direction = args_ptr->normalized_direction_;
        auto start_point_grid = args_ptr->start_point_grid_;
        auto end_point_grid = args_ptr->end_point_grid_;

        ScalarT denominator = args_ptr->denominator_;
        ScalarT multiplier_q = args_ptr->multiplier_q_;

        const std::size_t tid = args_ptr->thread_id_;
        const std::size_t subgrid_size = subgrid_storage.extent(0);
        const std::size_t standard_subgrid_size = args_ptr->standard_subgrid_size_;
        const std::size_t offset = tid * standard_subgrid_size;

        auto interval = end_point_grid - start_point_grid;
        ScalarT size_interval = interval.GetL2Norm();

        for (std::size_t i = 0; i < subgrid_size; i++) {
            //(x, y) = (x_0, y_0) + (V_x, V_y) * (b-a) * t уравнение прямой
            //t = (1 - r^i) / 1 - r^N
            subgrid_storage(i, 0) = //x
                start_point_grid.x + normalized_direction.x * size_interval *
                        (ScalarT(1) - fast_pow(multiplier_q, offset + i)) / denominator;
            subgrid_storage(i, 1) = //y
                start_point_grid.y + normalized_direction.y * size_interval *
                        (ScalarT(1) - fast_pow(multiplier_q, offset + i)) / denominator;
        }
        return nullptr;
    }

    template <kokkos_view_2d_like ContainerT, execution_policy Policy>
    struct GenNonUniformOnLine {
        using ScalarT = ContainerT::value_type;

        void operator() (
                    const Point2D<ScalarT> &normalized_direction,
                    const Point2D<ScalarT> &start_point_grid,
                    const Point2D<ScalarT> &end_point_grid,
                    ScalarT multiplier_q,
                    ContainerT grid_storage
                    ) const noexcept {
            std::size_t cpu_count;
            if constexpr (is_parallel<Policy>)
                cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
            else
                cpu_count = 1;

            const std::size_t grid_size = grid_storage.extent(0);
            const std::size_t subrange = grid_size / cpu_count; // Размер целого подотрезка
            const std::size_t tail_subrange = grid_size - (subrange) * cpu_count; //Остаток в последнем подотрезке
            const std::size_t last_subrange = subrange + tail_subrange; // Размер последнего подотрезка
            const ScalarT denominator = ScalarT(1) - std::pow(multiplier_q, grid_size - 1);

            std::vector<pthread_t> tid(cpu_count);
            using SubviewType = decltype(
                                    Kokkos::subview(
                                        grid_storage,
                                            Kokkos::pair<std::size_t, std::size_t>(0, 1),
                                            Kokkos::ALL
                                        )
                                    );
            std::vector<ThreadArgs<SubviewType>> args; // Обязательно выделять память не на стеке, через new или вектор,
            args.reserve(cpu_count);                   // Поскольку если оставить аргументы на стеке, то при выходе из области видимости незакончившие потоки будут ссылаться на мусор
            for(std::size_t t = 0; t < cpu_count; t++){
                pthread_attr_t attr;
                pthread_attr_init(&attr);
                cpu_set_t set;
                CPU_ZERO(&set);
                CPU_SET(t, &set);
                pthread_attr_setaffinity_np(&attr, sizeof(set), &set);
                std::size_t start_offset = t * subrange;
                auto grid_subrange = (t != cpu_count - 1) ?
                                            Kokkos::subview(
                                                    grid_storage,
                                                        Kokkos::pair(start_offset, start_offset + subrange),
                                                        Kokkos::ALL
                                                    ) // Стандартный подотрезок
                                            :
                                            Kokkos::subview(
                                                    grid_storage,
                                                        Kokkos::pair(start_offset, start_offset + last_subrange),
                                                        Kokkos::ALL
                                                        ); // Последний подотрезок

                ThreadArgs<SubviewType> arg(
                                        grid_subrange,
                                        normalized_direction,
                                        start_point_grid,
                                        end_point_grid,
                                        multiplier_q,
                                        denominator,
                                        subrange,
                                        t
                                    );
                args.push_back(std::move(arg));
                if (pthread_create(&tid[t], &attr, &worker<SubviewType>, &args[t]))
                    std::cerr << "cannot create thread: " << t << '\n';
            }
            for (std::size_t t = 0; t < cpu_count; t++)
                pthread_join(tid[t], nullptr);

        }

    };

    template <kokkos_view_2d_like ContainerT>
    struct GenUniformOnCircle {
        using ScalarT = ContainerT::value_type;

        void operator() (
                    ScalarT start_arc_angle_rad,
                    ScalarT end_arc_angle_rad,
                    ScalarT radius,
                    ContainerT grid_storage
                    ) const noexcept {
            const std::size_t grid_size = grid_storage.extent(0);
            const ScalarT step_on_circle = (end_arc_angle_rad - start_arc_angle_rad) / (grid_size - 1);
            for (std::size_t i = 0; i < grid_size; i++) {
                grid_storage(i, 0) = radius * std::cos(i * step_on_circle);
                grid_storage(i, 1) = radius * std::sin(i * step_on_circle);
            }

        }

    };


}

namespace mesh {

    template <kokkos_view_2d_like ContainerT>
    struct GenFrameKirsch {
        using ScalarT = ContainerT::value_type;

        void operator() (
                    ScalarT radius_hole,
                    ScalarT side_size,
                    ScalarT multiplier_q,
                    std::size_t count_nodes_on_hole,
                    std::size_t count_rays_in_each_sector,
                    ContainerT mesh_storage
                    ) const noexcept {


        }
    };

}