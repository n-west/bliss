
add_library(bland
    bland_tensor_internals.cpp
    ndarray.cpp
    ndarray_slice.cpp
    ndarray_deferred.cpp
    internal/include/internal/shape_helpers.cpp

    ops/ops.cpp
    ops/arithmetic.cpp
    ops/cpu/arithmetic_cpu.cpp
    ops/cpu/arithmetic_scalar_cpu.cpp
    ops/cpu/comparison_cpu.cpp
    ops/cpu/frequency_cpu_impl.cpp
    ops/cpu/ops_cpu.cpp
    ops/cpu/statistical_cpu.cpp
    ops/comparison.cpp
    ops/frequency.cpp
    ops/creators.cpp
    ops/statistical.cpp

    config.cpp
)

if (WITH_CUDA)
    target_sources(bland
        PRIVATE
        ops/cuda/arithmetic_cuda.cu
        ops/cuda/arithmetic_scalar_cuda.cu
        ops/cuda/comparison_cuda.cu
        ops/cuda/frequency_cuda.cu
        ops/cuda/ops_cuda.cu
        ops/cuda/statistical_cuda.cu
    )

    target_link_libraries(bland
        PRIVATE
        CUDA::cudart_static
        CUDA::cufft
    )
endif ()

target_include_directories(bland PRIVATE ${BLAND_GEN_INCLUDE_DIR})

target_include_directories(bland
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/include
)

target_link_libraries(bland
    PUBLIC
    dlpack_link
    PRIVATE
    fmt::fmt-header-only
    )


set_target_properties(bland PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

add_library(bland_internal INTERFACE)
target_include_directories(bland_internal
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/internal
)


add_executable(demo demo.cpp)
target_link_libraries(demo PUBLIC bland fmt::fmt-header-only)