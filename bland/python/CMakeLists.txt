
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
