
nanobind_add_module(pybliss pybliss.cpp)
target_link_libraries(pybliss
    PUBLIC
    pybliss_core
    pybland_header
    pydrift_search
    pyestimators
    pyfile_types
    pyflaggers
    PRIVATE
    fmt::fmt-header-only
)

if (WITH_CUDA)
    target_link_libraries(pybliss PRIVATE CUDA::cudart_static)
endif()

# Check if PY_BUILD_CMAKE_MODULE_NAME is defined
if(DEFINED PY_BUILD_CMAKE_MODULE_NAME)
    # Use the defined value
    set(PYBLISS_MODULE_INSTALL_DIR ${PY_BUILD_CMAKE_MODULE_NAME})
else()
    # Set a default value (e.g., install in site-packages)
    set(PYBLISS_MODULE_INSTALL_DIR "lib/python${PYTHON_VERSION}/site-packages")
endif()

install(TARGETS pybliss
        COMPONENT python_modules
        LIBRARY DESTINATION ${PYBLISS_MODULE_INSTALL_DIR})
