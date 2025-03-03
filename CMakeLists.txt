cmake_minimum_required(VERSION 3.23)
# Future version upgrades: 3.9 required for check_ipo_supported 3.14 adds FetchContent 3.18 has many cuda improvements
# 3.28 adds EXCLUDE_FROM_ALL to FetchContent_Declare

project(
  Bliss
  VERSION 0.0.1
  LANGUAGES CXX
)

include(CMakeDependentOption)
include(FeatureSummary)
# We don't actually need C, but there's an issue with debian packaging of cap'n proto that causes gcc to fail
# https://github.com/chaincodelabs/libmultiprocess/issues/68 The 2 solutions: * enable C * hard code that libatomic was
# found
enable_language(C)

include(CheckLanguage)
check_language(CUDA)

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
  message("CUDAToolkit_FOUND")
endif()

cmake_dependent_option(WITH_CUDA "Enable CUDA for bliss operations" ON "CMAKE_CUDA_COMPILER;CUDAToolkit_FOUND" OFF)
add_feature_info(CUDA WITH_CUDA "Enabling bliss-specific CUDA operations")

if(WITH_CUDA)
  set(CMAKE_CUDA_ARCHITECTURES 70)
  message("**BLISS:: enabling cuda (the language)")
  message("!!**: CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
  enable_language(CUDA)
  add_definitions("-DBLISS_CUDA=1")
else()
  message("**BLISS:: CUDA not found, building without it")
endif()

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/" ${CMAKE_MODULE_PATH})
#
# options
#

set(BUILD_SHARED_LIBS_DEFAULT OFF)

option(BUILD_SHARED_LIBS "BL: build shared libs" ${BUILD_SHARED_LIBS_DEFAULT})
option(BL_SANITIZE_THREAD "BL: enable thread sanitizer" OFF)
option(BL_SANITIZE_ADDRESS "BL: enable address sanitizer" OFF)
option(BL_SANITIZE_UNDEFINED "BL: enable undefined sanitizer" OFF)

#
# sanitizers
#

if(BL_SANITIZE_THREAD)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   -fsanitize=thread")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
endif()

if(BL_SANITIZE_ADDRESS)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   -fsanitize=address -fno-omit-frame-pointer")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
endif()

if(BL_SANITIZE_UNDEFINED)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   -fsanitize=undefined")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
endif()

# This is required for python modules...
set_property(GLOBAL PROPERTY POSITION_INDEPENDENT_CODE TRUE)

# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

  # Optionally set things like CMAKE_CXX_STANDARD, CMAKE_POSITION_INDEPENDENT_CODE here
  set(CMAKE_CXX_STANDARD 17) # Cmake 3.8
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  # Let's ensure -std=c++xx instead of -std=g++xx
  set(CMAKE_CXX_EXTENSIONS OFF)

  # Let's nicely support folders in IDEs
  set_property(GLOBAL PROPERTY USE_FOLDERS ON) # default in cmake 3.26+

  # # Testing only available if this is the main app # Note this needs to be done in the main CMakeLists # since it
  # calls enable_testing, which must be in the # main CMakeLists. include(CTest)

  # # Docs only available if this is the main app find_package(Doxygen) if(Doxygen_FOUND) add_subdirectory(docs) else()
  # message(STATUS "Doxygen not found, not building docs") endif()
endif()
include(FetchContent) # CMake 3.14+
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_AVAILABLE)

if(${IPO_AVAILABLE})
  message("**BLISS:: Can add flto")
endif()

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}") # CMake 3.9+
endif()

# HDF5 set(HDF5_USE_STATIC_LIBRARIES ON)
find_package(HDF5 REQUIRED COMPONENTS CXX)
set_package_properties(
  HDF5 PROPERTIES
  URL "https://www.hdfgroup.org/solutions/hdf5/"
  DESCRIPTION "Hierarchical Data Format"
  PURPOSE "Read/write BL HDF5-wrapped filterbank data"
)

# FetchContent_Declare(
#   external_Matx
#   EXCLUDE_FROM_ALL
#   GIT_REPOSITORY https://github.com/NVIDIA/MatX
#   GIT_TAG v0.9.0
# )
# FetchContent_MakeAvailable(external_Matx)

# Formatting library
FetchContent_Declare(
  fmtlib
  EXCLUDE_FROM_ALL
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 598e5180857954bf4e8f21f2a8c526691a071af1
)
FetchContent_MakeAvailable(fmtlib)
# Adds fmt::fmt

find_package(holoscan REQUIRED CONFIG
            PATHS "/opt/nvidia/holoscan/holoscan-linux-x86_64-2.8.0.1-archive" "/opt/nvidia/holoscan")

# Cap'n proto find_package(CapnProto QUIET) (disable until we fix using installed capnp)
if(NOT CapnProto_FOUND)
  # It's possible that the capnp executable exists without the library which will cause an error when their versions
  # inevitably don't match
  unset(CAPNP_EXECUTABLE)
  # capn proto requires CXX extensions
  # https://github.com/capnproto/capnproto/blob/a12c15a7cd0ae5121dca6216f529fb15fdee238c/c%2B%2B/CMakeLists.txt#L152C43-L152C63
  # TODO: try to scope these better
  set(CMAKE_CXX_EXTENSIONS ON)
  set(CMAKE_POSITION_INDEPENDENT_CODE
      ON
      CACHE BOOL "Enable position independent code"
  )
  message("**BLISS:: capn proto not found. Attempting to pull in as an external project (and enabling CXX extensions)")
  # Cap'n proto (de)serialization library
  FetchContent_Declare(
    external_capnproto
    EXCLUDE_FROM_ALL
    GIT_REPOSITORY https://github.com/capnproto/capnproto
    GIT_TAG v1.0.2
  )
  FetchContent_MakeAvailable(external_capnproto)
  set(CapnProto_FOUND TRUE)
endif()
set_package_properties(
  CapnProto PROPERTIES
  URL "https://capnproto.org/"
  DESCRIPTION "Serialization protocol with minimal encoding overhead"
  PURPOSE "Publishes hits and events in a structured way"
)

cmake_dependent_option(WITH_CAPNP "Enable Cap'n Proto to publish hits and events" ON "CapnProto_FOUND" OFF)
add_feature_info(WITH_CAPNP CapnProto_FOUND "CAPNP serialization of hits and events")


find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
# nanobind (for python bindings)
FetchContent_Declare(
  nanobind
  EXCLUDE_FROM_ALL
  GIT_REPOSITORY https://github.com/wjakob/nanobind
  GIT_TAG v1.7.0
)
FetchContent_MakeAvailable(nanobind)

# The compiled library code is here
add_subdirectory(bliss)
add_subdirectory(bland)

# Find Doxygen package
find_package(Doxygen)
find_package(Sphinx QUIET)

# Check if Doxygen is installed
cmake_dependent_option(BUILD_DOCS "Build docs (doxygen + sphinx/breathe)" ON "DOXYGEN_FOUND;SPHINX_FOUND" OFF)
add_feature_info(DOCUMENTATION BUILD_DOCS "doxygen and sphinx based documentation")

if(BUILD_DOCS)
  add_subdirectory(docs)
else()
  message("Doxygen needs to be installed to generate the doxygen documentation")
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  feature_summary(WHAT ENABLED_FEATURES DISABLED_FEATURES PACKAGES_FOUND)
  feature_summary(FILENAME ${CMAKE_CURRENT_BINARY_DIR}/features.log WHAT ALL)
endif()
