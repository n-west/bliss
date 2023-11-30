cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(
    bland
    VERSION 0.0.0
    LANGUAGES CXX
)

# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

    set(CMAKE_CXX_STANDARD 17) # Cmake 3.8
    set(CMAKE_CXX_STANDARD_REQUIRED On)
    set(CMAKE_CXX_EXTENSIONS Off)

endif()


# This is where ubuntu installs cmake config files. TODO: there's got to be a better way...
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    list(APPEND CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/cmake")
endif()


## dlpack
# We need minimum 0.8.0 for DLManagedTensorVersioned
find_package(dlpack 0.8.0 QUIET)

if (dl_pack_FOUND)
message("[[BLAND]] Using system version of dlpack.")
    set(dlpack_link dlpack::dlpack)
else ()
    message("[[BLAND]] Couldn't find a system version of dlpack with minimum version 0.7.0. Using our own header.")
    add_library(dlpack_link INTERFACE)
    # needs to be BEFORE just in case there's an older system header installed
    target_include_directories(dlpack_link BEFORE INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external/dlpack)
endif()
## - end dlpack - ##


add_subdirectory(bland)
add_subdirectory(python)
add_subdirectory(tests)

