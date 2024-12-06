
Install
=======

BLISS Dedrift is primarily written in C++ and CUDA C++ with python bindings through nanobind. The build system uses
cmake to build the C++ shared libraries and executables. There is a python build system that can be invoked with pip
to build a python package. The major build artifacts include:

* a C++ API with shared library and public headers
* python API
* executable utilities for running basic pipelines and generating pipeline artifacts
* documentation

Pre-built binary packages are available

Pre-built packages (Conda)
------------------

Conda packages are available from anaconda.org that support multiple versions of python and cuda:

.. csv-table:: Support for python and cuda versions
   :file: python-cuda-conda-matrix

These packages include the python API as well as executable scripts that can run hit search pipelines, event search
pipelines, and generate artifacts like polyphase filterbank responses for various telescope configurations.

```
conda install -c nwest blissdedrift
```


From source
-----------

Building from source should only be necessary for development of BLISS dedrift itself which is primarily with cmake.
This is relatively straight-forward with a few system-specific caveats for systems that you might be using. First,
we'll cover the general case then focus on system-specific issues later

Build-time dependencies
~~~~~~~~~~~~~~~~~~~~~~~

The following requirements must be met to build bliss dedrift from source:

* a C++-17 compiler
* cmake >= 3.28
* fftw (we can probably make this optional!)
* libhdf5-cpp

At runtime, you'll also very likely need bitshuffle and hdf5-filter-plugin for working with most Breakthrough Listen
data.

CMake
~~~~~

You can use cmake to build everything. This is mostly good for development. The standard process is to create a build
directory, tell cmake where the source code is, and build.

.. code-block::

    mkdir build
    cd build
    cmake ..
    make -j $(($(nproc)/2))


Python / PEP 517 build
~~~~~~~~~~~~~~~~~~~~~~

Alternatively you can use the PEP 517-compatible wrapper around cmake and python packaging. The ``pyproject.toml`` file
fully describes the build so you can use ``pip install`` or other PEP 517 frontends like ``python -m build``


Following PEP 517, the ``pyproject.toml``. describes 
You can use pip to build a wheel which has python bindings. You can do this with a checked out copy of the source code
by running

```
pip install -e .
```

from within the source directory. The ``-e`` argument does an "editable" install which will allow you to make some
changes and rebuild without going through the install procedure again or allow changes to the python-specific parts
of the package without another install.

Conda-based psuedo cross-compile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conda is often used to install packages that are missing or require an upgrade from what is available from the host
system. This can include fftw, hdf5, or even an entire compiler.

Psuedo-cross compiling (working on a system with an old compiler that doesn't support C++-17)
#############################################################################################

You can build bliss on older systems that don't support C++-17 (Ubuntu 16.04 as an example) by using conda to install
a newer compiler. Installing a compiler through conda is a little complicated because it requires what's known as a
psuedo cross-compile (see https://docs.conda.io/projects/conda-build/en/latest/resources/compiler-tools.html#an-aside-on-cmake-and-sysroots).
Bliss comes with a cmake `toolchain` file to help with this. First, install the newer compiler:

``conda install gxx=11.4``

The other quirk of treating the build as a cross-compile is that you cannot depend and link against system libraries
on your host system. Every dependency has to be installed in the conda environment, so you'll also need to use conda
to install fftw, hdf5, bitshuffle (this isn't a build dependency that's linked in, but installing it in conda makes
the conda-provided hdf5 able to find it).

```
conda install hdf5 bitshuffle hdf5-external-filter-plugins fftw
```

```
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/conda.cmake ..
make -j $(($(nproc)/2))
```


Data center specific tips and quirks
------------------------------------

You can check what distribution and version you're running with `lsb_release -a`.


Berkeley DC (Ubuntu 24.04)
~~~~~~~~~~~~~~~~~~~~~~~

The hosts running Ubuntu 24.04 have everything required, so no conda dependencies are needed unless you want to use
conda dependencies. If you are using the installed cuda 11.7.1, then the host system's gcc (version 13.2.0) is not
compatible, so you won't be able to build cuda kernels with that GCC. You'll need to set the following variable before
running cmake:

```
export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++-11'
```

The specific gcc version you point to may be subject to change, but a table of cuda versions to maximum supported gcc version
can be found at https://stackoverflow.com/a/46380601


Berkeley DC (Ubuntu 16.04)
~~~~~~~~~~~~~~~~~~~~~~~

Some nodes are running Ubuntu 16.06 (blpc1 at time of this writing) which needs to follow the psuedo cross-compile instructions
above. The whole process looks like this:

```
# Source the cuda environment
. /usr/local/cuda-11.7.1/cuda.sh

# Create and activate conda environment
conda create -n nwest-build-blpc1 python=3.10 gxx=11.4 cmake fftw hdf5 bitshuffle hdf5-external-filter-plugins fftw # This can take a while to solve the environment
conda activate nwest-build-blpc1

# Build with the psuedo-cross toolchain for conda
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/conda.cmake ..
make -j $(($(nproc)/2))
```

MacOS
~~~~~

I haven't tried and don't have the hardware to try, but you'll need to make sure any rosetta issues with dependencies don't exist.

