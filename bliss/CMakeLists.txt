

add_subdirectory(core)
add_subdirectory(estimators)
add_subdirectory(file_types)
add_subdirectory(flaggers)
add_subdirectory(rfi)
add_subdirectory(drift_search)

add_subdirectory(tests)
add_subdirectory(python)

add_executable(bliss_find_hits bliss_find_hits.cpp)

target_link_libraries(bliss_find_hits
    bliss_core
    drift_search
    estimators
    bliss_serialization
    flaggers
    fmt::fmt-header-only
    )
    
add_executable(event_search event_search.cpp)
target_link_libraries(event_search
    bliss_serialization
    drift_search
    fmt::fmt-header-only
    )
