

add_library(bland
    bland_tensor_internals.cpp
    ndarray.cpp
    ops/ops.cpp
    ops/arithmetic.cpp
    ops/arithmetic_scalar.cpp
    ops/comparison.cpp
    ops/creators.cpp
    ops/statistical.cpp
    internal/shape_helpers.cpp
)

target_include_directories(bland
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/internal
)

target_precompile_headers(bland
  PRIVATE
    [["bland/bland.hpp"]]
    [["bland/ndarray.hpp"]]
    [["bland/ops.hpp"]]
    [["dispatcher.hpp"]]
    [["elementwise_binary_op.hpp"]]
    [["elementwise_scalar_op.hpp"]]
    [["elementwise_unary_op.hpp"]]
)

target_link_libraries(bland
    PUBLIC
    dlpack_link
    PRIVATE
    fmt::fmt-header-only
    )

if (CUDAToolkit_FOUND)
    target_link_libraries(bland
    PRIVATE
    CUDA::cudart_static)
endif()

set_target_properties(bland PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

add_library(bland_internal INTERFACE)
target_include_directories(bland_internal
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/internal
)
