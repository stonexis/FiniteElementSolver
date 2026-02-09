#pragma once
#include <Kokkos_Core.hpp>
#include "core/custom_concepts.hpp"
#include "core/geometry/geometry.hpp"
#include "solutions/custom_pthreads/grid/grid.hpp"

namespace mesh {

    /**
     * Функция выпуска луча в 2D пространстве из точки zero_point, проходящий через точку на отверстии hole_storage[idx_point_on_hole]
     * со значениями от hole_point до точки пересечения с границей пластины
     * @tparam ContainerT
     * @tparam ScalarT
     * @tparam PolicyFillRay Вызов функции GenNonUniformOnRay (Parallel/Sequential)
     * @param zero_point
     * @param hole_point
     * @param first_point_edge Граница задается двумя точками (first_point_edge, second_point_edge)
     * @param second_point_edge Граница задается двумя точками (first_point_edge, second_point_edge)
     * @param multiplier_q
     * @param ray_storage
     */
    template <kokkos_view_2d_like ContainerT, execution_policy PolicyFillRay, typename ScalarT>
    void emit_ray(
                const geometry::Point2D<ScalarT> &zero_point,
                const geometry::Point2D<ScalarT> &hole_point,
                const geometry::Point2D<ScalarT> &first_point_edge,
                const geometry::Point2D<ScalarT> &second_point_edge,
                ScalarT multiplier_q,
                ContainerT ray_storage
                ) noexcept;

    template <kokkos_view_2d_like ContainerT, execution_policy PolicyEmitRays>
    struct GenFrameKirsch {
        using ScalarT = ContainerT::value_type;
        /**
         * Генерация каркаса сетки
         * Каркас сетки имеет вид | ___ | ___ | ___ | , где | - главные лучи, разделяющие сетку на сектора по потокам. _ - внутренние лучи, генерируемые внутри сектора потоком
         * @param pthreads_pool Менеджер потоков
         * @param radius_hole
         * @param side_size Размер стороны пластины (пластина квадратная)
         * @param multiplier_q Основание геометрической прогрессии для роста интервала между точками для сторон пластины, прилежащих к отверстию
         * @param count_points_on_hole Требуемое общее количество точек на отверстии (count_points_on_hole - 1) % threads == 0
         * @param count_points_on_ray Количество точек на луче
         * @return Kokkos::View<double*[2], Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Restrict | Kokkos::Aligned>>; (ViewType)
         */
        [[nodiscard]] ViewType operator() (
                        pthreads_manage::Pool &pthreads_pool,
                        ScalarT radius_hole,
                        ScalarT side_size,
                        ScalarT multiplier_q,
                        std::size_t count_points_on_hole, // (count_points_on_hole - 1) % threads == 0
                        std::size_t count_points_on_ray
                        ) const noexcept;
    };

}

#include "mesh_impl.tpp"
