

add_subdirectory(core)
add_subdirectory(estimators)
add_subdirectory(file_types)
add_subdirectory(flaggers)
add_subdirectory(preprocess)
add_subdirectory(rfi)
add_subdirectory(drift_search)

add_subdirectory(tests)
add_subdirectory(python)

add_executable(bliss_generate_channelizer_response bliss_generate_channelizer_response.cpp)
target_link_libraries(bliss_generate_channelizer_response
    bliss_preprocess
    bland
    fmt::fmt-header-only
    )

add_executable(bliss_find_hits bliss_find_hits.cpp)
target_link_libraries(bliss_find_hits
    bliss_core
    drift_search
    estimators
    bliss_serialization
    flaggers
    fmt::fmt-header-only
    )

add_executable(bliss_hits_to_dat bliss_hits_to_dat.cpp)
target_link_libraries(bliss_hits_to_dat
    bliss_serialization
    fmt::fmt-header-only
    )

add_executable(event_search event_search.cpp)
target_link_libraries(event_search
    bliss_serialization
    drift_search
    fmt::fmt-header-only
    )
