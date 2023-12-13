
cmake_minimum_required(VERSION 3.3.25.1)
# Future version upgrades:
# 3.28 adds EXCLUDE_FROM_ALL to FetchContent_Declare


project(
    Bliss
    VERSION 0.0.0
    LANGUAGES CXX
)

set_property(GLOBAL PROPERTY POSITION_INDEPENDENT_CODE TRUE)

# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

  # Optionally set things like CMAKE_CXX_STANDARD, CMAKE_POSITION_INDEPENDENT_CODE here
  set(CMAKE_CXX_STANDARD 17) # Cmake 3.8
  set(CMAKE_CXX_STANDARD_REQUIRED On)
  set(CMAKE_CXX_EXTENSIONS Off)

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

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache) # Less useful to do it for linking, see edit2
endif(CCACHE_FOUND)


find_package(HDF5 COMPONENTS CXX REQUIRED)

include(FetchContent) # CMake 3.14+

# Formatting library
FetchContent_Declare(
  fmtlib
  EXCLUDE_FROM_ALL
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 5.3.0)
FetchContent_MakeAvailable(fmtlib)
# Adds fmt::fmt

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
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
