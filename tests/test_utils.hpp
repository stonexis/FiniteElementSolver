#pragma once
#include "../src/include.hpp"
#include <gtest/gtest.h>
#include <random>


template<std::size_t N>
struct S_Single {
    static constexpr std::size_t N_ = N;
};

template<std::size_t N, std::size_t M>
struct S_Two {
    static constexpr std::size_t N_ = N;
    static constexpr std::size_t M_ = M;
};

using ViewType = Kokkos::View<
                            double*[2],
                            Kokkos::LayoutRight,
                            Kokkos::HostSpace,
                            Kokkos::MemoryTraits<Kokkos::Restrict | Kokkos::Aligned>
                            >;

template<typename S>
class SingleGridFixture : public ::testing::Test{
public:
    static constexpr std::size_t N = S::N_;
    ViewType grid;

    void SetUp() override {
        auto alloc = Kokkos::view_alloc(Kokkos::WithoutInitializing, "v");
        grid = ViewType(alloc,N);

    }

    void TearDown() override {}

};

template<typename S>
class TwoGridFixture : public ::testing::Test{
public:
    static constexpr std::size_t N = S::N_;
    static constexpr std::size_t M = S::M_;

    ViewType grid_first;
    ViewType grid_second;

    void SetUp() override {
        auto alloc = Kokkos::view_alloc(Kokkos::WithoutInitializing, "v");
        grid_first = ViewType(alloc,N);
        grid_second = ViewType(alloc,M);

    }

    void TearDown() override {}

};




using AllSizesSingle = ::testing::Types<
                            S_Single<10>,
                            S_Single<3>,
                            S_Single<4>,
                            S_Single<50>,
                            S_Single<31>,
                            S_Single<153>,
                            S_Single<7>,
                            S_Single<8>,
                            S_Single<5>,
                            S_Single<500>
                        >;
using AllSizesTwo = ::testing::Types<
                            S_Two<10, 10>,
                            S_Two<3, 3>,
                            S_Two<4, 5>,
                            S_Two<50, 59>,
                            S_Two<31, 28>,
                            S_Two<153, 45>,
                            S_Two<7, 2>,
                            S_Two<500, 600>
                        >;

TYPED_TEST_SUITE(SingleGridFixture, AllSizesSingle);
TYPED_TEST_SUITE(TwoGridFixture, AllSizesTwo);
