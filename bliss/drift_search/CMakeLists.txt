
add_library(drift_search
    connected_components.cpp
    integrate_drifts.cpp
    event_search.cpp
    filter_hits.cpp
    hit_search.cpp
    local_maxima.cpp
)

target_include_directories(drift_search
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(drift_search
    PUBLIC
    bland
    bliss_core
    PRIVATE
    fmt::fmt-header-only
    flaggers
)

set_target_properties(drift_search PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

add_library(pydrift_search INTERFACE)
target_link_libraries(pydrift_search INTERFACE drift_search)

