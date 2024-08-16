
add_library(file_types
    h5_filterbank_file.cpp
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

add_library(bliss_serialization
    dat_file.cpp
    hits_file.cpp
)
target_include_directories(bliss_serialization
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
target_link_libraries(bliss_serialization
    PRIVATE
    fmt::fmt-header-only
)

# This is optional and only makes sense if capnp is available
if (WITH_CAPNP)
    target_compile_definitions(bliss_serialization PUBLIC -DBLISS_USE_CAPNP=1)
    # TODO: condition out some includes so actual usage of capnp is disabled as well
    target_sources(bliss_serialization
        PRIVATE
        cpnp_files.cpp
        events_file.cpp
        detail/raii_file_helpers.cpp
        detail/cpnp_hit_builder.cpp
    )
    target_link_libraries(bliss_serialization
        PUBLIC
        bliss_core
        PRIVATE
        CapnProto::capnp
    )
    capnp_generate_cpp(CAPNP_SRCS CAPNP_HDRS hit.capnp)
    message("** Generated CAPNP_SRCS: ${CAPNP_SRCS}")
    target_sources(bliss_serialization PRIVATE ${CAPNP_SRCS})
    target_include_directories(bliss_serialization PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

    # Add a target eventually to do this:
    # capnp compile hit.capnp --output=c++:include/file_types/
    # Generates
    # * include/file_types/hit.capnp.h
    # * hit.capnp.c++
    # The generated header needs to match the library version used
    # seticore uses sumbodule w/ version 11000
    # ubuntu 23.04 packages 9002
endif()


set_target_properties(file_types PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(bliss_serialization PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

add_library(pyfile_types INTERFACE)
target_link_libraries(pyfile_types INTERFACE file_types bliss_serialization)
