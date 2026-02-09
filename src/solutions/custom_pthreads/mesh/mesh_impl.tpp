#pragma once

namespace mesh {
    template <typename ScalarT>
    struct KernelArgsEmitRay {
        geometry::Point2D<ScalarT> zero_point_;
        geometry::Point2D<ScalarT> first_point_right_edge_, first_point_up_edge_, second_point_edge_;
        ScalarT multiplier_q_;
        std::size_t count_sectors_;
        ViewType hole_storage_;
    };

    struct PartitionerArgs {
        std::size_t full_size_;
        std::size_t count_points_on_ray_;
        std::size_t count_rays_between_masters_;
        std::size_t count_sectors_;
    };
    ///Разделение по принципу 1 сектор = 1 поток
    inline pthreads_manage::PartitionerSettings partitioner(void* args) {
        auto* args_ptr = static_cast<PartitionerArgs*>(args);
        std::size_t count_points_on_ray = args_ptr->count_points_on_ray_;
        std::size_t count_rays_between_masters = args_ptr->count_rays_between_masters_;
        //Сетка имеет вид: |---|---|
        //Внутреннее незаполненное пространство для последующего заполнения сетки без учета краевых лучей сектора
        std::size_t size_inner_space = count_rays_between_masters * count_points_on_ray;
        std::size_t size_entire_sector = size_inner_space + 2 * count_points_on_ray; // | --- |
        return pthreads_manage::PartitionerSettings{
                                            args_ptr->full_size_,
                                 size_entire_sector,
                                            args_ptr->count_points_on_ray_, // Пересечение двух секторов = крайний луч
                                        };

    }

    template <kokkos_view_2d_like ContainerT>
    void* threadDispatch(ViewType ray_storage, std::size_t tid, void* args){
        using ScalarT = ContainerT::value_type;
        using p_type = geometry::Point2D<ScalarT>;
        auto* args_ptr = static_cast<KernelArgsEmitRay<ScalarT>*>(args);
        //Начальная точка границы, с которой ищем пересечение зависит от сектора из которого выпускаем луч (т.е. номера потока)
        auto first_point_edge = (tid >= args_ptr->count_sectors_ /2) ? args_ptr->first_point_up_edge_ : args_ptr->first_point_right_edge_;
        emit_ray<Sequential>( // Заполняем сетку последовательно, поскольку доступных потоков нет
            args_ptr->zero_point,
            p_type{args_ptr->hole_storage_(tid, 0), args_ptr->hole_grid_tmp(tid, 1)},
            first_point_edge,
            args_ptr->second_point_edge,
            args_ptr->multiplier_q,
            ray_storage
            );
        return nullptr;
    }

    template <kokkos_view_2d_like ContainerT, execution_policy PolicyEmitRays>
    ViewType GenFrameKirsch<ContainerT, PolicyEmitRays>::operator() (
                                pthreads_manage::Pool &pthreads_pool,
                                ScalarT radius_hole,
                                ScalarT side_size,
                                ScalarT multiplier_q,
                                std::size_t count_points_on_hole, // (count_points_on_hole - 1) % threads == 0
                                std::size_t count_points_on_ray
                                ) const noexcept {
        std::size_t count_sectors;
        if constexpr (is_parallel<PolicyEmitRays>)
            count_sectors = pthreads_pool.totalThreads();
        else
            count_sectors = 1;

        std::size_t count_master_rays_total = count_sectors + 1;
        std::size_t count_slave_rays_total = count_points_on_hole - count_master_rays_total;
        std::size_t count_slaves_between_masters = count_master_rays_total / count_sectors;
        std::size_t mesh_size = (count_master_rays_total + count_slave_rays_total) * count_points_on_ray;

        auto alloc = Kokkos::view_alloc(Kokkos::WithoutInitializing, "v");
        auto mesh_storage = ViewType(alloc, mesh_size);

        //Временная сетка для отверстия, для стартовой генерации. В итоговой сетке точки на окружности будут автоматически из за первой точки лучей
        auto hole_grid_tmp = ViewType(alloc, count_master_rays_total);

        kernels::fill_circle_arc_uniform(0.0, std::numbers::pi/2.0, radius_hole, hole_grid_tmp);

        using p_type = geometry::Point2D<ScalarT>;
        p_type zero_point{ScalarT(0.0), ScalarT(0.0)}; // Все лучи выпускаются из точки (0,0)
        p_type first_point_right_edge{side_size, ScalarT(0.0)};
        p_type first_point_up_edge{ScalarT(0.0), side_size};
        p_type second_point_edge{side_size, side_size}; // Вторая точка у обоих границ общая


        PartitionerArgs part_args{mesh_size, count_points_on_ray, count_slaves_between_masters, count_sectors};
        auto part_args_ptr = std::make_unique<PartitionerArgs>(part_args);
        KernelArgsEmitRay<ScalarT> kernel_args{
                                        zero_point,
                                        first_point_right_edge,
                                        first_point_up_edge,
                                        second_point_edge,
                                        multiplier_q,
                                        count_sectors,
                                        hole_grid_tmp
                                    };
        auto kernel_args_ptr = std::make_unique<KernelArgsEmitRay<ScalarT>>(kernel_args);
        pthreads_manage::JobContext context{
                                mesh_storage,
                                &threadDispatch,
                                kernel_args_ptr.get(),
                                        &partitioner,
                                part_args_ptr.get()
                                };

        //Запускаем генерацию лучей параллельно
        pthreads_pool.dispatchJob(context);

        //Отдельно заполняем последний луч (вертикальный) тк он остался необработанный
        auto last_ray = Kokkos::subview(
                                mesh_storage,
                                Kokkos::pair(mesh_size - count_points_on_ray, mesh_size - 1),
                                Kokkos::ALL
                            );
        emit_ray(
            zero_point,
            p_type{hole_grid_tmp(count_master_rays_total - 1, 0), hole_grid_tmp(count_master_rays_total - 1, 1)},
            first_point_up_edge,
            second_point_edge,
            multiplier_q,
            last_ray
            );

        return mesh_storage;
    }

    template <kokkos_view_2d_like ContainerT, execution_policy PolicyFillRay, typename ScalarT>
    void emit_ray(
                const geometry::Point2D<ScalarT> &zero_point,
                const geometry::Point2D<ScalarT> &hole_point,
                const geometry::Point2D<ScalarT> &first_point_edge,
                const geometry::Point2D<ScalarT> &second_point_edge,
                ScalarT multiplier_q,
                ContainerT ray_storage
                ) noexcept {
        using p_type = geometry::Point2D<ScalarT>;
        p_type interception_point = geometry::interception_lines(
                                                zero_point,
                                                hole_point,
                                                first_point_edge,
                                                second_point_edge
                                                        );
        // Так как первая точка нулевая, то точка на окружности = вектор направления луча
        p_type direction = hole_point;
        direction.Normalize();

        grid::GenNonUniformOnRay<ScalarT, PolicyFillRay> gen_ray;
        gen_ray(
            direction,
            hole_point,
            interception_point,
            multiplier_q,
            ray_storage
            );
    }

}