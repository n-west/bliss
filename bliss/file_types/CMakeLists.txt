
add_library(file_types
    h5_filterbank_file.cpp
    hits_file.cpp
)

target_include_directories(file_types
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(file_types
    PUBLIC
    bland
    HDF5::HDF5
    PRIVATE
    fmt::fmt-header-only
)

target_link_libraries(file_types
    PRIVATE
    ${CapnProto_LIBRARIES}
)

# Add a target eventually to do this:
# capnp compile hit.capnp --output=c++:include/file_types/
# Generates
# * include/file_types/hit.capnp.h
# * hit.capnp.c++
# The generated header needs to match the library version used
# seticore uses sumbodule w/ version 11000
# ubuntu 23.04 packages 9002

set_target_properties(file_types PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

add_library(pyfile_types INTERFACE)
target_link_libraries(pyfile_types INTERFACE file_types)
