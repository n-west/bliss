

add_subdirectory(convenience)
add_subdirectory(estimators)
add_subdirectory(file_types)
add_subdirectory(flaggers)
add_subdirectory(rfi)
add_subdirectory(spectrumsum)

add_subdirectory(tests)
add_subdirectory(python)

add_executable(justrun justrun.cpp)
target_link_libraries(justrun
    spectrumsum
    file_types
    estimators
    HDF5::HDF5
    fmt::fmt
    )


