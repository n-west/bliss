
add_library(bliss_core
    cadence.cpp
    scan.cpp
    filterbank_data.cpp
    noise_power.cpp
)

target_include_directories(bliss_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bliss_core
    PRIVATE
    file_types
    fmt::fmt-header-only
)

set_target_properties(bliss_core PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

add_library(pybliss_core INTERFACE)
target_link_libraries(pybliss_core INTERFACE bliss_core)
