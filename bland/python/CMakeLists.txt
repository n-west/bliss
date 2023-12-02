
# find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
# # nanobind (for python bindings)
# FetchContent_Declare(
#   nanobind
#   GIT_REPOSITORY https://github.com/wjakob/nanobind
#   GIT_TAG v1.7.0)
# FetchContent_MakeAvailable(nanobind)


add_library(pybland_header
    INTERFACE
)
target_include_directories(pybland_header
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

nanobind_add_module(pybland pybland.cpp)
target_link_libraries(pybland PUBLIC bland)

target_compile_options(pybland
    PRIVATE
        -O3
)
