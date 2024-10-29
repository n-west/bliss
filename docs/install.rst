
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
