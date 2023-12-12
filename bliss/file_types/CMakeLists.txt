
add_library(file_types
    h5_filterbank_file.cpp
    filterbank_data.cpp
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

set_target_properties(file_types PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

add_library(pyfile_types INTERFACE)
# target_include_directories()
target_link_libraries(pyfile_types INTERFACE file_types)