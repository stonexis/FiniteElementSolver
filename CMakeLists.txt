cmake_minimum_required(VERSION 4.0)
project(FEM LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 75;80;86)


include(FetchContent)

#--- Kokkos ---
FetchContent_Declare(
        Kokkos
        URL https://github.com/kokkos/kokkos/archive/refs/tags/5.0.0.zip
)
FetchContent_MakeAvailable(Kokkos)

#--- onemkl ---
if (DEFINED ENV{MKLROOT})
    set(MKLROOT $ENV{MKLROOT})
else()
    set(MKLROOT "/opt/intel/oneapi/mkl/latest")
endif()

list(APPEND CMAKE_PREFIX_PATH
        "${MKLROOT}/lib/cmake/mkl"
        "/opt/intel/oneapi/tbb/latest/lib/cmake/tbb")

find_package(MKL CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)

set(SRC
    #src/solutions/custom_cuda/
    src/solutions/custom_pthreads/mesh/pthreads_impl.cpp
    #src/solutions/custom_sequential/
)

add_executable(${PROJECT_NAME} main.cpp ${SRC})

target_link_libraries(${PROJECT_NAME} PRIVATE MKL::MKL TBB::tbb Kokkos::kokkos)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ----- Google Test -----
include(CTest)
if (BUILD_TESTING)

    set(INSTALL_GTEST OFF)
    set(gtest_build_tests OFF)
    FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/heads/main.zip
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    add_executable(fem_tests
            ${SRC}
            tests/test_main.cpp
            tests/test_utils.cpp
    )

    target_include_directories(fem_tests
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )

    target_link_libraries(fem_tests
            PRIVATE
            GTest::gtest_main
            MKL::MKL
            TBB::tbb
            Kokkos::kokkos
    )

    include(GoogleTest)
    gtest_discover_tests(fem_tests
            PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    )#add_test find source file, discover_tests - find binary
endif()

# ----- Google Benchmark -----
option(BUILD_CPU_BENCHMARKS "Build CPU microbenchmarks")
if (BUILD_BENCHMARKS)

    set(BENCHMARK_ENABLE_TESTING      OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS  OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL      OFF CACHE BOOL "" FORCE)
    # set(BENCHMARK_ENABLE_LIBPFM    ON  CACHE BOOL "" FORCE)

    FetchContent_Declare(
            benchmark
            URL https://github.com/google/benchmark/archive/refs/heads/main.zip
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(benchmark)
    find_package(Threads REQUIRED)

    add_executable(fem_benchmarks
            ${SRC}
            benchmarks/
    )

    target_include_directories(fem_benchmarks
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
    )

    target_link_libraries(fem_benchmarks
            PRIVATE
            benchmark::benchmark
            Threads::Threads
            MKL::MKL
            TBB::tbb
            Kokkos::kokkos
    )

endif()

#----- nvbenchmark ----
option(BUILD_GPU_BENCHMARKS "Build GPU microbenchmarks")
if (BUILD_GPU_BENCHMARKS)

    FetchContent_Declare(
            nvbench
            GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
            GIT_TAG main
    )
    FetchContent_MakeAvailable(nvbench)

    add_executable(fem_nvbench
            ${SRC}
            benchmarks/
    )

    target_include_directories(fem_nvbench PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
    )

    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(fem_nvbench
            PRIVATE
            nvbench
            CUDA::cudart
            Kokkos::kokkos
    )

endif()
