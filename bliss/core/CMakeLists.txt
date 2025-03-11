
add_library(bliss_core
    cadence.cpp
    coarse_channel.cpp
    event.cpp
    frequency_drift_plane.cpp
    hit.cpp
    noise_power.cpp
    scan.cpp
)

target_include_directories(bliss_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bliss_core
    PUBLIC
    bland
    PRIVATE
    file_types
    fmt::fmt-header-only
)

set_target_properties(bliss_core PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

add_library(pybliss_core INTERFACE)
target_link_libraries(pybliss_core INTERFACE bliss_core)

add_library(holobliss_core
    holoscan_scan.cpp
)
target_link_libraries(holobliss_core PUBLIC CCCL::CCCL holoscan::core PRIVATE bliss_core)