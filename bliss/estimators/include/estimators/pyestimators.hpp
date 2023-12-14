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


    nb::enum_<bliss::noise_power_estimator>(m, "noise_power_estimator")
    .value("stddev", bliss::noise_power_estimator::STDDEV)
    .value("mad", bliss::noise_power_estimator::MEAN_ABSOLUTE_DEVIATION);
    
    nb::class_<bliss::noise_power_estimate_options>(m, "noise_power_estimate_options")
    .def(nb::init<>())
    .def_rw("estimator_method", &bliss::noise_power_estimate_options::estimator_method)
    .def_rw("masked_estimate", &bliss::noise_power_estimate_options::masked_estimate);

    m.def("estimate_noise_power", nb::overload_cast<const bland::ndarray&, bliss::noise_power_estimate_options>(&bliss::estimate_noise_power));
    m.def("estimate_noise_power", nb::overload_cast<bliss::filterbank_data, bliss::noise_power_estimate_options>(&bliss::estimate_noise_power));
    m.def("estimate_noise_power", [](nb::ndarray<> arr, bliss::noise_power_estimate_options opts) {
        return bliss::estimate_noise_power(nb_to_bland(arr), opts);
    });

    nb::class_<bliss::noise_stats>(m, "noise_power")
    .def_prop_ro("noise_floor", &bliss::noise_stats::noise_floor)
    .def_prop_ro("noise_amplitude", &bliss::noise_stats::noise_amplitude)
    .def_prop_ro("noise_power", &bliss::noise_stats::noise_power);


}
