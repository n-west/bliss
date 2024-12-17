
add_library(bliss_preprocess
    excise_dc.cpp
    normalize.cpp
    passband_static_equalize.cpp
)

target_include_directories(bliss_preprocess
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(bliss_preprocess
    PUBLIC
    bland
    bliss_core
    PRIVATE
    fmt::fmt-header-only
)


set_target_properties(bliss_preprocess PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

add_library(pybliss_preprocess INTERFACE)
# target_include_directories()
target_link_libraries(pybliss_preprocess INTERFACE bliss_preprocess)
