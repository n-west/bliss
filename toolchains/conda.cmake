
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

set(CMAKE_SYSROOT $ENV{CONDA_PREFIX}/x86_64-conda-linux-gnu/sysroot)
# where is the target environment
set(CMAKE_FIND_ROOT_PATH $ENV{CONDA_PREFIX} $ENV{CONDA_PREFIX}/x86_64-conda-linux-gnu/sysroot $ENV{CUDA_DIR})

# search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# god-awful hack because it seems to not run correct tests to determine this:
set(__CHAR_UNSIGNED___EXITCODE 1)

