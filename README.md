
# Breakthrough Listen Interesting Signal Search

<p align="center">

[![Build](https://github.com/n-west/bliss/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/n-west/bliss/actions/workflows/build-and-test.yml)

</p>


## Building and Experimenting

This project builds with cmake. Building and running depend on the following libraries/tools:

### build:

* cmake
* gcc / clang capable of C++17
* libhdf5-dev


### runtime:

* libhdf5-cpp-103 (double check!)
* hdf5-filter-plugin
* bitshuffle


### Building

The build system uses cmake with a few dependencies listed below. I recommend building as follows (from the project source folder)

```
mkdir build
cd build
cmake .. # -G Ninja # if you prefer ninja
make -j $(($(nproc)/2)) # replace with make -j CORES if you don't have nproc
```

Inside `build/bland/tests` you will have a `test_bland` executable. You can run those tests to sanity check everything works as expected.

Inside `build/bliss/` you will have a `justrun` executable. You can pass a fil file to that, but right now this is a useful debugging and sanity check target, the output will be underwhelming.

Inside `build/bliss/python` you should have a `pybliss.cpython-311-x86_64-linux-gnu.so` or similarly named `pybliss` shared library. This can be imported in python and exposes functions and classes for dedoppler searches. Inside the `notebooks` directory there is an `rfi mitigation visual.ipynb` jupyter notebook that walks through several of the functions with plots showing results. That is the best way to get a feel for functionality.

#### Python-only

(...this is a work in progress and doesn't work super well yet... Battling the ol' setuptools.)

The easiest/less path-fiddly way to get the python extensions running is to run `setup.py develop`. This will (should) build the extension module and install it in "editable mode". I suggest setting up a virtualenv for this project, then doing this. The result should be pybliss in your python path ready to run.

`setup.py develop` # A normal build, will take a few minutes but run faster


If you have setuptools 49.2.0 or later, supposedly something like this would work... (this is a TODO to make fiddling with this easier)
```
python setup.py develop --config-setting="build_ext --debug"
```

The library should now be available with `from bliss import pybliss`. I'd like to fix this to avoid to nested module before release.

When you're done, run python `setup.py develop --uninstall`


### Optimizations

The backing compute library, bland, is set up for flexibility in running different "ops" on opaquely typed and shaped `ndarray` type objects. This uses C++ templating pretty heavily to define what the operation is mostly independent of how to do broadcasting and indexing to traverse ndarrays which may have strides and arbitrary shapes. This use of templating generates a lot of code which all benefits greatly from compiler optimizations. Turning on a `RelWithDebInfo` or `Release` build will make everything run much faster than a `Debug` build, but also takes several more minutes to compile.

Tips:
 * I nearly always build and develop in Debug mode because it's much faster to build and I can wait the moment longer for results
 * I build Release builds when profiling and tuning optimizations
 * We'll build binary distributions with Release build types

Additionally, the cuda backend is not (yet) in place. That's a near-top priority over the next week or two once all algorithmic and API choices are beginning to settle.
