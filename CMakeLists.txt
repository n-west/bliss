
cmake_minimum_required(VERSION 3.9)
# Future version upgrades:
# 3.9 required for check_ipo_supported
# 3.14 adds FetchContent
# 3.18 has many cuda improvements
# 3.28 adds EXCLUDE_FROM_ALL to FetchContent_Declare


project(
    Bliss
    VERSION 0.0.0
    LANGUAGES CXX
)

include(CMakeDependentOption)
include(FeatureSummary)
# We don't actually need C, but there's an issue with debian packaging of cap'n proto
# that causes gcc to fail
# https://github.com/chaincodelabs/libmultiprocess/issues/68
# The 2 solutions:
# * enable C
# * hard code that libatomic was found
enable_language(C)

include(CheckLanguage)
check_language(CUDA)

find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
    message("CUDAToolkit_FOUND")
endif()

cmake_dependent_option(WITH_CUDA "Enable CUDA for bliss operations" ON "CMAKE_CUDA_COMPILER;CUDAToolkit_FOUND" OFF)
add_feature_info(CUDA WITH_CUDA "Enabling bliss-specific CUDA operations")

if (WITH_CUDA)
  message("**BLISS:: enabling cuda (the language)")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  add_definitions("-DBLISS_CUDA=1")
else()
  message("**BLISS:: CUDA not found, building without it")
endif()

##
# options
##

set(BUILD_SHARED_LIBS_DEFAULT OFF)

option(BUILD_SHARED_LIBS          "BL: build shared libs" ${BUILD_SHARED_LIBS_DEFAULT})
option(BL_SANITIZE_THREAD         "BL: enable thread sanitizer"    OFF)
option(BL_SANITIZE_ADDRESS        "BL: enable address sanitizer"   OFF)
option(BL_SANITIZE_UNDEFINED      "BL: enable undefined sanitizer" OFF)

##
# sanitizers
##

if (BL_SANITIZE_THREAD)
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -fsanitize=thread")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
endif()

if (BL_SANITIZE_ADDRESS)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}     -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
endif()

if (BL_SANITIZE_UNDEFINED)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}     -fsanitize=undefined")
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

#   # Testing only available if this is the main app
#   # Note this needs to be done in the main CMakeLists
#   # since it calls enable_testing, which must be in the
#   # main CMakeLists.
#   include(CTest)

#   # Docs only available if this is the main app
#   find_package(Doxygen)
#   if(Doxygen_FOUND)
#     add_subdirectory(docs)
#   else()
#     message(STATUS "Doxygen not found, not building docs")
#   endif()
endif()
include(FetchContent) # CMake 3.14+
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_AVAILABLE)

if (${IPO_AVAILABLE})
  message("**BLISS:: Can add flto")
endif()

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}") # CMake 3.9+
endif()


# HDF5
find_package(HDF5 QUIET REQUIRED COMPONENTS CXX)
set_package_properties(HDF5 PROPERTIES
    URL "https://www.hdfgroup.org/solutions/hdf5/"
    DESCRIPTION "Hierarchical Data Format"
    PURPOSE "Read/write BL HDF5-wrapped filterbank data")

# Cap'n proto
find_package(CapnProto QUIET)
if (NOT CapnProto_FOUND)
  # capn proto requires CXX extensions
  # https://github.com/capnproto/capnproto/blob/a12c15a7cd0ae5121dca6216f529fb15fdee238c/c%2B%2B/CMakeLists.txt#L152C43-L152C63
  # TODO: try to scope these better
  set(CMAKE_CXX_EXTENSIONS ON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Enable position independent code")
  message("**BLISS:: capn proto not found. Attempting to pull in as an external project (and enabling CXX extensions)")
  # Cap'n proto (de)serialization library
  FetchContent_Declare(
    external_capnproto
    EXCLUDE_FROM_ALL
    GIT_REPOSITORY https://github.com/capnproto/capnproto
    GIT_TAG v1.0.2)
  FetchContent_MakeAvailable(external_capnproto)
  set(CapnProto_FOUND TRUE)
endif ()
set_package_properties(CapnProto PROPERTIES
    URL "https://capnproto.org/"
    DESCRIPTION "Serialization protocol with minimal encoding overhead"
    PURPOSE "Publishes hits and events in a structured way")

cmake_dependent_option(WITH_CAPNP "Enable Cap'n Proto to publish hits and events" ON "CapnProto_FOUND" OFF)
add_feature_info(WITH_CAPNP CapnProto_FOUND "CAPNP serialization of hits and events")

# Formatting library
FetchContent_Declare(
  fmtlib
  EXCLUDE_FROM_ALL
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 5.3.0)
FetchContent_MakeAvailable(fmtlib)
# Adds fmt::fmt

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
# nanobind (for python bindings)
FetchContent_Declare(
  nanobind
  EXCLUDE_FROM_ALL
  GIT_REPOSITORY https://github.com/wjakob/nanobind
  GIT_TAG v1.7.0)
FetchContent_MakeAvailable(nanobind)


# The compiled library code is here
add_subdirectory(bliss)
add_subdirectory(bland)

# Find Doxygen package
find_package(Doxygen)

# Check if Doxygen is installed
if (DOXYGEN_FOUND)
    set(DOXYGEN_INPUT_DIR ${CMAKE_SOURCE_DIR})
    # Set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    # Request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Doxygen build started")

    # Note the option ALL which allows to build the docs together with the application
    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
    set_target_properties(doc PROPERTIES EXCLUDE_FROM_ALL TRUE)
else (DOXYGEN_FOUND)
  message("Doxygen needs to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)


if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    feature_summary(WHAT ENABLED_FEATURES DISABLED_FEATURES PACKAGES_FOUND)
    feature_summary(FILENAME ${CMAKE_CURRENT_BINARY_DIR}/features.log WHAT ALL)
endif()
