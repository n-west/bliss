#pragma once

#include "noise_estimate.hpp"
#include "spectral_kurtosis.hpp"
#include <core/noise_power.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

void bind_pyestimators(nb::module_ m) {

    m.def("estimate_spectral_kurtosis", &bliss::estimate_spectral_kurtosis, "spectrum_grid"_a, "N"_a, "M"_a, "d"_a = 1,
        "Compute spectral kurtosis of the given spectra"
    );

    m.def("estimate_noise_power", nb::overload_cast<const bland::ndarray&, bliss::noise_power_estimate_options>(&bliss::estimate_noise_power));
    m.def("estimate_noise_power", nb::overload_cast<bliss::filterbank_data, bliss::noise_power_estimate_options>(&bliss::estimate_noise_power));

}
