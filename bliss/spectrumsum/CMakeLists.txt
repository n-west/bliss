
add_library(spectrumsum
    spectrumsum.cpp
    hit_search.cpp
)

target_include_directories(spectrumsum
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(spectrumsum
    PUBLIC
    bland
    PRIVATE
    fmt::fmt-header-only
)

set_target_properties(spectrumsum PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

nanobind_add_module(pyspectrumsum pyspectrumsum.cpp)
target_link_libraries(pyspectrumsum PRIVATE spectrumsum)
