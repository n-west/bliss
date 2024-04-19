
# Breakthrough Listen Interesting Signal Search

<p align="center">

[![Build](https://github.com/n-west/bliss/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/n-west/bliss/actions/workflows/build-and-test.yml)

</p>

![alien teaching signals](docs/alien-teaching-signals.jpeg)

BLISS is a toolkit for finding narrowband doppler-drifting signals. This is frequently used to search for [technosignatures](https://en.wikipedia.org/wiki/Technosignature). 

BLISS is able to use cuda-accelerated kernels with deferred execution and memoization for flagging, noise estimation, integration, and hit search.

## Installation

Running bliss requires

* libhdf5-cpp-103 (double check!)
* hdf5-filter-plugin
* bitshuffle

### Binary package

Prebuilt wheels are available for the following runtimes:

* **cpu only**: `pip install dedrift`
* **cuda 11**: `pip install dedrift-cuda11x`
* **cuda 12**: `pip install dedrift-cuda12x`


### Building from source

This project builds with cmake and is set up to be built as a python package with `pyproject.toml`. Building and running depend on the following libraries/tools:

* cmake
* gcc / clang capable of C++17 and supporting your version of cuda
* libhdf5-dev
* (optional) libcapnp-dev

#### CMake-based (dev) builds:

The standard cmake workflow should configure everything for a build:
```
mkdir build
cd build
cmake .. # -G Ninja # if you prefer ninja
make -j $(($(nproc)/2)) # replace with make -j CORES if you don't have nproc
```

The python package is partially set up in `bliss/python/bliss`. During the build process, the C++ extensions are built and placed in this package. In cmake development mode, this is placed in build/bliss/python/bliss and configured to be updated with any file changes as they occur (each file is symlinked) and new files will be added at the next build.


#### Python package build

`pyproject.toml` configures the python package and uses `py-cmake-build` as the build backend to get the required C++ extensions built and packaged appropriately. You can build this package with standard tools such as `pip install .` and `python -m build`.


## Tests
Inside `build/bland/tests` you will have a `test_bland` executable. You can run those tests to sanity check everything works as expected.

Inside `build/bliss/` you will have a `justrun` executable. You can pass a fil file to that, but right now this is a useful debugging and sanity check target, the output will be underwhelming.

Inside `build/bliss/python` you should have a `pybliss.cpython-311-x86_64-linux-gnu.so` or similarly named `pybliss` shared library. This can be imported in python and exposes functions and classes for dedoppler searches. Inside the `notebooks` directory there is an `rfi mitigation visual.ipynb` jupyter notebook that walks through several of the functions with plots showing results. That is the best way to get a feel for functionality.


## Usage

### Python

The following is example usage for Voayger-1 recordings from the Green Bank Telescope

```python
import bliss

data_loc = "/datag/public/voyager_2020/single_coarse_channel/old_single_coarse/"
cadence = bliss.cadence([[f"{data_loc}/single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5",
                    f"{data_loc}/single_coarse_guppi_59046_80672_DIAG_VOYAGER-1_0013.rawspec.0000.h5",
                    f"{data_loc}/single_coarse_guppi_59046_81310_DIAG_VOYAGER-1_0015.rawspec.0000.h5"
                    ],
                    [f"{data_loc}/single_coarse_guppi_59046_80354_DIAG_VOYAGER-1_0012.rawspec.0000.h5"],
                    [f"{data_loc}/single_coarse_guppi_59046_80989_DIAG_VOYAGER-1_0014.rawspec.0000.h5"],
                    [f"{data_loc}/single_coarse_guppi_59046_81628_DIAG_VOYAGER-1_0016.rawspec.0000.h5"]])


cadence.set_device("cuda:0")

working_cadence = cadence
working_cadence = bliss.flaggers.flag_filter_rolloff(working_cadence, .2)
working_cadence = bliss.flaggers.flag_spectral_kurtosis(working_cadence, .05, 25)


noise_est_options = bliss.estimators.noise_power_estimate_options()
noise_est_options.masked_estimate = True
noise_est_options.estimator_method = bliss.estimators.noise_power_estimator.stddev

working_cadence = bliss.estimators.estimate_noise_power(working_cadence, noise_est_options)

int_options = bliss.integrate_drifts_options()
int_options.desmear = True
int_options.low_rate = -500
int_options.high_rate = 500

working_cadence = bliss.drift_search.integrate_drifts(working_cadence, int_options)

working_cadence.set_device("cpu")

hit_options = bliss.drift_search.hit_search_options()
hit_options.snr_threshold = 10
cadence_with_hits = bliss.drift_search.hit_search(working_cadence, hit_options)

hits_dict = bliss.plot_utils.get_hits_list(cadence_with_hits)
```

### C++

```c++
    auto voyager_cadence = bliss::cadence({{"/datag/public/voyager_2020/single_coarse_channel/old_single_coarse/single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5",
                    "/datag/public/voyager_2020/single_coarse_channel/old_single_coarse/single_coarse_guppi_59046_80672_DIAG_VOYAGER-1_0013.rawspec.0000.h5",
                    "/datag/public/voyager_2020/single_coarse_channel/old_single_coarse/single_coarse_guppi_59046_81310_DIAG_VOYAGER-1_0015.rawspec.0000.h5"
                    },
                    {"/datag/public/voyager_2020/single_coarse_channel/old_single_coarse/single_coarse_guppi_59046_80354_DIAG_VOYAGER-1_0012.rawspec.0000.h5"},
                    {"/datag/public/voyager_2020/single_coarse_channel/old_single_coarse/single_coarse_guppi_59046_80989_DIAG_VOYAGER-1_0014.rawspec.0000.h5"},
                    {"/datag/public/voyager_2020/single_coarse_channel/old_single_coarse/single_coarse_guppi_59046_81628_DIAG_VOYAGER-1_0016.rawspec.0000.h5"}});

    auto cadence = voyager_cadence;

    cadence.set_device("cuda:0");

    cadence = bliss::flag_filter_rolloff(cadence, 0.2);
    cadence = bliss::flag_spectral_kurtosis(cadence, 0.1, 25);

    cadence = bliss::estimate_noise_power(
            cadence,
            bliss::noise_power_estimate_options{.estimator_method=bliss::noise_power_estimator::STDDEV, .masked_estimate = true}); // estimate noise power of unflagged data

    cadence = bliss::integrate_drifts(
            cadence,
            bliss::integrate_drifts_options{.desmear        = true,
                                            .low_rate       = -500,
                                            .high_rate      = 500,
                                            .rate_step_size = 1});

    cadence.set_device("cpu");

    auto cadence_with_hits = bliss::hit_search(cadence, {.method=bliss::hit_search_methods::CONNECTED_COMPONENTS,
                                                        .snr_threshold=10.0f});

    auto events = bliss::event_search(cadence);

    bliss::write_events_to_file(events, "events_output");
```