
# This script is meant to be invoked by conda-build and depends on the following
# env vars from conda-build (https://docs.conda.io/projects/conda-build/en/latest/user-guide/environment-variables.html):
# SRC_DIR - the location of source tree (this file should live in SRC_DIR/conda-recipe/build.sh
# PREFIX - the location conda expects us to install to
# CPU_COUNT - number of cpus returned by python multiprocessing.cpu_count()
# MAKEFLAGS - forwarded from host environmetn


mkdir build-conda
cd build-conda
cmake \
    ${CMAKE_ARGS} \
    ${SRC_DIR} \
    -DCMAKE_INSTALL_PREFIX=${PREFIX}
make -j 4 ${MAKEFLAGS}
make install

