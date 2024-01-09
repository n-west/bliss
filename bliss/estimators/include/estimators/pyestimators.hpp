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

    m.def("estimate_spectral_kurtosis",
          nb::overload_cast<const bland::ndarray &, int64_t, int64_t, float>(&bliss::estimate_spectral_kurtosis),
          "spectrum_grid"_a,
          "N"_a,
          "M"_a,
          "d"_a = 1,
          "Compute spectral kurtosis of the given spectra");
    m.def("estimate_spectral_kurtosis",
          nb::overload_cast<bliss::filterbank_data &>(&bliss::estimate_spectral_kurtosis),
          "fil_data"_a,
          "Compute spectral kurtosis of the given filterbank data");

    nb::enum_<bliss::noise_power_estimator>(m, "noise_power_estimator")
            .value("stddev", bliss::noise_power_estimator::STDDEV)
            .value("mad", bliss::noise_power_estimator::MEAN_ABSOLUTE_DEVIATION);

    nb::class_<bliss::noise_power_estimate_options>(m, "noise_power_estimate_options")
            .def(nb::init<>())
            .def_rw("estimator_method", &bliss::noise_power_estimate_options::estimator_method)
            .def_rw("masked_estimate", &bliss::noise_power_estimate_options::masked_estimate);

    m.def("add", [](nb::ndarray<> a, nb::ndarray<> b) {
        return bland::add(nb_to_bland(a), nb_to_bland(b));
    });

    m.def("estimate_noise_power_dbg", [](nb::ndarray<> arr) {
        auto opts = bliss::noise_power_estimate_options();
        return bliss::estimate_noise_power(nb_to_bland(arr), opts);
    });

    m.def("estimate_noise_power", [](nb::ndarray<> arr, bliss::noise_power_estimate_options opts) {
        return bliss::estimate_noise_power(nb_to_bland(arr), opts);
    });
    m.def("estimate_noise_power",
          nb::overload_cast<const bland::ndarray &, bliss::noise_power_estimate_options>(&bliss::estimate_noise_power));
    m.def("estimate_noise_power",
          nb::overload_cast<bliss::filterbank_data, bliss::noise_power_estimate_options>(&bliss::estimate_noise_power));
    m.def("estimate_noise_power",
          nb::overload_cast<bliss::observation_target, bliss::noise_power_estimate_options>(
                  &bliss::estimate_noise_power));
    m.def("estimate_noise_power",
          nb::overload_cast<bliss::cadence, bliss::noise_power_estimate_options>(&bliss::estimate_noise_power));

    nb::class_<bliss::noise_stats>(m, "noise_power")
            .def_prop_ro("noise_floor", &bliss::noise_stats::noise_floor)
            .def_prop_ro("noise_amplitude", &bliss::noise_stats::noise_amplitude)
            .def_prop_ro("noise_power", &bliss::noise_stats::noise_power);
}
