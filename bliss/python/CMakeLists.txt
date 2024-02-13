
nanobind_add_module(pybliss pybliss.cpp)
target_link_libraries(pybliss
    PUBLIC
    pybliss_core
    pybland_header
    pydrift_search
    pyestimators
    pyfile_types
    pyflaggers
)

install(TARGETS pybliss LIBRARY DESTINATION pybliss)

