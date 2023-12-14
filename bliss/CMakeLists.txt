

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


