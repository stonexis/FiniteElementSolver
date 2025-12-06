#pragma once
#include <type_traits>
#include <concepts>

struct Sequential {};
struct Parallel {};

template<class P>
concept point2d_like = requires(P p){
    {p.x} -> std::floating_point;
    {p.y} -> std::floating_point;
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