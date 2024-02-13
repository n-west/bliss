
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

# Check if PY_BUILD_CMAKE_MODULE_NAME is defined
if(DEFINED PY_BUILD_CMAKE_MODULE_NAME)
    # Use the defined value
    set(PYBLISS_MODULE_INSTALL_DIR ${PY_BUILD_CMAKE_MODULE_NAME})
else()
    # Set a default value (e.g., install in site-packages)
    set(PYBLISS_MODULE_INSTALL_DIR "lib/python${PYTHON_VERSION}/site-packages")
endif()

install(TARGETS pybland
        COMPONENT python_modules
        LIBRARY DESTINATION ${PYBLISS_MODULE_INSTALL_DIR})
