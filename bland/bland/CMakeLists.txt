

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
target_compile_options(bland
    PRIVATE
        -Ofast
        -march=native
        -mtune=native
)

# find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
# # nanobind (for python bindings)
# FetchContent_Declare(
#   nanobind
#   GIT_REPOSITORY https://github.com/wjakob/nanobind
#   GIT_TAG v1.7.0)
# FetchContent_MakeAvailable(nanobind)


nanobind_add_module(pybland pybland.cpp)
target_link_libraries(pybland PUBLIC bland)

target_compile_options(pybland
    PRIVATE
        -O3
)

