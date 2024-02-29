
Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.3.0 # or a later release
)

FetchContent_MakeAvailable(Catch2)

# TODO: we might actually want to split benchmarks to their own executable
add_executable(test_bland
    test.cpp
    benchmark_arithmetic.cpp
    benchmark_mean.cpp
    test_arithmetic.cpp
    test_comparison.cpp
    test_copy.cpp
    test_creators.cpp
    test_ndarray.cpp
    test_strides.cpp
    test_reductions.cpp
    test_statistical.cpp
    )

target_link_libraries(test_bland
    PRIVATE
    bland
    Catch2::Catch2
    fmt::fmt-header-only
    )

if (WITH_CUDA)
    target_link_libraries(test_bland
        PRIVATE
        CUDA::cudart_static
    )
endif()

# # Helps auto discovery of test reporting to ctest
# list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
# include(CTest)
# include(Catch)
# catch_discover_tests(test_bland)
