

add_library(bland_internal INTERFACE
    internal/elementwise_binary_op.hpp
)

target_include_directories(bland_internal
    INTERFACE
    include
)


add_library(bland
    ndarray.cpp
    ops/ops.cpp
    ops/arithmetic.cpp
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

target_link_libraries(bland
    PUBLIC
    dlpack_link
    PRIVATE
    fmt::fmt-header-only
    )

set_target_properties(bland PROPERTIES POSITION_INDEPENDENT_CODE TRUE)


