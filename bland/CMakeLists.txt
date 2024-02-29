cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(
    bland
    VERSION 0.0.0
    LANGUAGES CXX
)
include(CheckLanguage)
check_language(CUDA)

find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
    message("CUDAToolkit_FOUND")
endif()

cmake_dependent_option(WITH_CUDA "Enable CUDA backend for array storage and operations" ON "CMAKE_CUDA_COMPILER;CUDAToolkit_FOUND" OFF)
add_feature_info(CUDA WITH_CUDA "Using CUDA as a backend for array storage and compute")

if (WITH_CUDA)
  message("BLAND:: enabling cuda (the language)")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  add_definitions("-DBLAND_CUDA_CODE=1")
  add_definitions("-DBLAND_CUDA=1")
else()
  message("BLAND:: CUDA not found, building without it")
endif()



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

