
# Breakthrough Listen Interesting Signal Search

<p align="center">

[![Build](https://github.com/n-west/bliss/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/n-west/bliss/actions/workflows/build-and-test.yml)

</p>

![alien teaching signals](docs/alien-teaching-signals.jpeg)

BLISS is a toolkit for finding narrowband doppler-drifting signals. This is frequently used to search for [technosignatures](https://en.wikipedia.org/wiki/Technosignature). 

BLISS is able to use cuda-accelerated kernels with deferred execution and memoization for flagging, noise estimation, integration, and hit search.

## Installation

See more detailed documentation in docs/install.rst which is rendered at https://bliss.readthedocs.io/en/latest/install.html

Building bliss requires:
* libhdf5-cpp-103
* libfftw-dev
* cmake
* a compiler capable of c++-17

Running bliss requires

* libhdf5-cpp-103 (double check!)
* hdf5-filter-plugin
* bitshuffle
* libfftw



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