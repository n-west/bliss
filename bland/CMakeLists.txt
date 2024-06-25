cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

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

  execute_process(COMMAND nvcc --list-gpu-code OUTPUT_VARIABLE NVCC_LIST_GPU_CODE)
  string(REPLACE "\n" ";" NVCC_CODE_LINES ${NVCC_LIST_GPU_CODE})

  # Process each line
  set(NVCC_ARCHS "")
  foreach(LINE ${NVCC_CODE_LINES})
      # Match the architecture in the line
      string(REGEX MATCH "sm_([0-9]+)" MATCH ${LINE})
      if(MATCH)
          # TODO: what happens when we get to major arch 10
          string(SUBSTRING ${CMAKE_MATCH_1} 0 1 MAJOR)
          string(SUBSTRING ${CMAKE_MATCH_1} 1 2 MINOR)
          set(ARCH "${MAJOR}${MINOR}")
          list(APPEND NVCC_ARCHS ${ARCH})
      endif()
  endforeach()

  list(LENGTH NVCC_ARCHS NVCC_NUM_ARCHS)
  # Convert the list of architectures to a string
  string(JOIN "," NVCC_ARCHS_STR ${NVCC_ARCHS})

  # Custom command to generate bland_cuda_archs.h
  set(BLAND_GEN_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/bland/include/)
  set(GEN_CUDA_ARCHS_HPP ${BLAND_GEN_INCLUDE_DIR}/generated/bland_cuda_archs.hpp)

  message("BLAND:: enabling cuda (the language)")
  set(CUDA_ARCHITECTURES all)
  enable_language(CUDA)
  add_definitions("-DBLAND_CUDA_CODE=1")
  add_definitions("-DBLAND_CUDA=1")
else()
  message("BLAND:: CUDA not found, building without it")
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/bland_cuda_archs.hpp.in ${GEN_CUDA_ARCHS_HPP} @ONLY)


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

