
nanobind_add_module(pybliss pybliss.cpp)
target_link_libraries(pybliss
    PUBLIC
    file_types
    spectrumsum
)

