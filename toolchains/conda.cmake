
##
# this toolchain is heavily influenced by https://docs.conda.io/projects/conda-build/en/latest/resources/compiler-tools.html#an-aside-on-cmake-and-sysroots
# which points to https://github.com/AnacondaRecipes/libnetcdf-feedstock/tree/master/recipe
# for a definitive conda toolchain file (see cross-linux.cmake)
#
# Working on ubuntu 16.04 netboot images with the following env seems to work with this toolchain file:
# conda config --add channels conda-forge
# conda install gxx=11.4.0 cmake hdf5 bitshuffle hdf5-external-filter-plugins fftw
#
# The really important parts are changing the find root for libraries and includes to use the conda env
# and disallow the host. The host system programs are fine though.
# Be careful to also include cuda in the find root path. CUDA_DIR is set through the /usr/bin/cuda-*/cuda.sh script
##

# specify the cross compiler
set(CMAKE_C_COMPILER $ENV{CC})

# Check if both CUDA_ROOT and CUDA_DIR are defined
if(DEFINED ENV{CUDA_ROOT} AND DEFINED ENV{CUDA_DIR})
    if(NOT "$ENV{CUDA_ROOT}" STREQUAL "$ENV{CUDA_DIR}")
        message(FATAL_ERROR "CUDA_ROOT and CUDA_DIR are both defined but different. CUDA_ROOT: $ENV{CUDA_ROOT}, CUDA_DIR: $ENV{CUDA_DIR}")
    else()
        set(CUDA_PATH $ENV{CUDA_ROOT})
    endif()
elseif(DEFINED ENV{CUDA_ROOT})
    set(CUDA_PATH $ENV{CUDA_ROOT})
elseif(DEFINED ENV{CUDA_DIR})
    set(CUDA_PATH $ENV{CUDA_DIR})
else()
    message(FATAL_ERROR "Neither CUDA_ROOT nor CUDA_DIR is defined.")
endif()

# Set the CUDA_PATH variable
set(CUDA_PATH ${CUDA_PATH} CACHE PATH "Path to CUDA")

set(CMAKE_SYSROOT $ENV{CONDA_PREFIX}/x86_64-conda-linux-gnu/sysroot)
# where is the target environment
set(CMAKE_FIND_ROOT_PATH $ENV{CONDA_PREFIX} $ENV{CONDA_PREFIX}/x86_64-conda-linux-gnu/sysroot ${CUDA_PATH})

# search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# god-awful hack because it seems to not run correct tests to determine this:
set(__CHAR_UNSIGNED___EXITCODE 1)

