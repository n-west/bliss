
add_library(rfi
    rfi.cpp
)

target_include_directories(rfi
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(rfi
    PUBLIC
    bland
    PRIVATE
    bland_internal
    fmt::fmt-header-only
)

# set_target_properties(spectrumsum PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

# nanobind_add_module(pyspectrumsum pyspectrumsum.cpp)
# target_link_libraries(pyspectrumsum PRIVATE spectrumsum)
