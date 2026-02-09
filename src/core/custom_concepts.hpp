#pragma once
#include <type_traits>
#include <concepts>
#include <Kokkos_Core.hpp>

using ViewType = Kokkos::View<double*[2], Kokkos::LayoutRight, Kokkos::HostSpace>;

struct Sequential {};
struct Parallel {};

template<class T>
concept scalar = std::floating_point<T>;

template<class P>
concept point2d_like = requires(P p){
    {p.x} -> scalar;
    {p.y} -> scalar;
};

template<class View>
concept kokkos_view_like = Kokkos::is_view_v<std::remove_cvref_t<View>>;

template<class View>
concept kokkos_view_rank2_like = kokkos_view_like<View> && std::remove_cvref_t<View>::rank == 2;

template<class View>
concept kokkos_view_2d_like = kokkos_view_rank2_like<View> && std::remove_cvref_t<View>::traits::dimension::N1 == 2;

template<class P>
concept is_parallel = std::same_as<std::remove_cvref_t<P>, Parallel>;

template<class P>
concept is_sequential = std::same_as<std::remove_cvref_t<P>, Sequential>;

template<class P>
concept execution_policy = is_sequential<P> || is_parallel<P>;

template<class E>
concept environment = requires (E e){
    e.run_kernel;
    e.kernel_args;

    e.partitioner;
    e.partitioner_args;
};


