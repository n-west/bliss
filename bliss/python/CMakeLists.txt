
nanobind_add_module(pybliss pybliss.cpp)
target_link_libraries(pybliss
    PUBLIC
    pybland_header
    pyestimators
    pyfile_types
    pyflaggers
    spectrumsum
)

