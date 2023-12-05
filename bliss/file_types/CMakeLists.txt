
add_library(file_types
    h5_filterbank_file.cpp
)

target_include_directories(file_types
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(file_types
    PUBLIC
    # datatypedefs
    bland
    HDF5::HDF5
    PRIVATE
    fmt::fmt-header-only
)
