#pragma once
#include <iostream>
#include <Kokkos_Core.hpp>
#include "core/custom_concepts.hpp"
#include "core/geometry/geometry.hpp"
#include "core/kernels/kernels.hpp"
#include "solutions/custom_pthreads/pthreads_manage.hpp"

namespace grid {

    template <kokkos_view_2d_like ContainerT, execution_policy Policy>
    class GenNonUniformOnRay {
        using ScalarT = ContainerT::value_type;
    public:
        /**
         * Функция для генерации сетки на луче на основе геометрической прогрессии
         * (x, y) = (x_0, y_0) + (V_x, V_y) * (b-a) * t уравнение прямой, где t = (1 - r^i) / 1 - r^N
         * @param pthreads_pool Менеджер потоков
         * @param normalized_direction_ray Вектор направления луча
         * @param start_point_grid Точка на луче, с которой начинается заполнение сетки (должна лежать на направляющем векторе)
         * @param end_point_grid Точка на луче, на которой заканчивается заполнение сетки (должна лежать на направляющем векторе)
         * @param multiplier_q Основание геометрической прогрессии для роста сетки
         * @param ray_storage
         */
        void operator() (
                    pthreads_manage::Pool &pthreads_pool,
                    const geometry::Point2D<ScalarT> &normalized_direction_ray,
                    const geometry::Point2D<ScalarT> &start_point_grid,
                    const geometry::Point2D<ScalarT> &end_point_grid,
                    ScalarT multiplier_q,
                    ContainerT ray_storage
                    ) const noexcept {
            std::size_t count_threads;
            if constexpr (is_parallel<Policy>)
                count_threads = pthreads_pool.totalThreads();
            else
                count_threads = 1;

            std::size_t grid_size = ray_storage.extent(0);
            const ScalarT denominator = ScalarT(1) - std::pow(multiplier_q, grid_size - 1);

            PartitionerArgs partitioner_args{grid_size, count_threads};
            auto partitioner_args_ptr = std::make_unique<PartitionerArgs>(partitioner_args);
            auto settings = partitioner(partitioner_args_ptr.get()); // Запускаем здесь разделитель тоже, поскольку нужен chunk_size для работы ядра (для индекса от начала)

            KernelArgs<ScalarT> kernel_args{
                                normalized_direction_ray,
                                start_point_grid,
                                end_point_grid,
                                multiplier_q,
                                denominator,
                                settings.chunk_size_
                            };
            auto kernel_args_ptr = std::make_unique<KernelArgs<ScalarT>>(kernel_args); // Обязательно выделять память не на стеке, через new или вектор,
            // Поскольку если оставить аргументы на стеке, то при выходе из области видимости незакончившие потоки будут ссылаться на мусор


            pthreads_manage::JobContext context{
                                    ray_storage,
                                    &threadDispatch,
                                    kernel_args_ptr.get(),
                                            &partitioner,
                                    partitioner_args_ptr.get()
                                    };
            //Запуск ядер
            pthreads_pool.dispatchJob(context);

        }
    private:
        template <typename ScalarT>
        struct KernelArgs {
            geometry::Point2D<ScalarT> normalized_direction_ray_;
            geometry::Point2D<ScalarT> start_point_full_grid_, end_point_full_grid_;
            ScalarT multiplier_q_;
            ScalarT denominator_;
            std::size_t chunk_size_;
        };
        ///Прослойка для распаковки параметров и запуска ядра
        static void threadDispatch(ViewType subrange, std::size_t worker_id, void* args) noexcept {
            auto* args_ptr = static_cast<KernelArgs<ScalarT>*>(args);

            auto normalized_direction_ray = args_ptr->normalized_direction_ray_;
            auto start_point_grid = args_ptr->start_point_full_grid_;
            auto end_point_grid = args_ptr->end_point_full_grid_;
            auto multiplier_q = args_ptr->multiplier_q_;
            auto denominator = args_ptr->denominator_;
            auto idx_from_start = args_ptr->chunk_size_ * worker_id;

            kernels::fill_ray_segment_nonuniform(
                            normalized_direction_ray,
                            start_point_grid,
                            end_point_grid,
                            multiplier_q,
                            denominator,
                            idx_from_start,
                            subrange
                            );
        }

        struct PartitionerArgs {
            std::size_t full_size_;
            std::size_t count_threads_;
        };
        ///Политика разделения на сегменты
        [[nodiscard]] static pthreads_manage::PartitionerSettings partitioner(void* args) noexcept {
            auto* args_ptr = static_cast<PartitionerArgs*>(args);
            std::size_t full_size = args_ptr->full_size_;
            std::size_t chunk_size = (full_size + args_ptr->count_threads_ - 1) / args_ptr->count_threads_; // Размер целого подотрезка

            return pthreads_manage::PartitionerSettings{full_size, chunk_size, 0};
        }
    };
}
