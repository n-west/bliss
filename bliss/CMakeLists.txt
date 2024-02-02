

add_subdirectory(core)
add_subdirectory(estimators)
add_subdirectory(file_types)
add_subdirectory(flaggers)
add_subdirectory(rfi)
add_subdirectory(drift_search)

add_subdirectory(tests)
add_subdirectory(python)

add_executable(justrun justrun.cpp)
target_link_libraries(justrun
    estimators
    flaggers
    file_types
    drift_search
    HDF5::HDF5
    fmt::fmt-header-only
    )

add_executable(read_hits read_hits.cpp)
target_link_libraries(read_hits
    file_types
    fmt::fmt-header-only
    )
    
add_executable(event_search event_search.cpp)
target_link_libraries(event_search
    file_types
    drift_search
    HDF5::HDF5
    fmt::fmt-header-only
    )
    