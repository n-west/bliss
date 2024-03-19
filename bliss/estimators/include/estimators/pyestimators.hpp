#pragma once

#include "noise_estimate.hpp"
#include "spectral_kurtosis.hpp"
#include <core/noise_power.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <fmt/format.h>

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
          nb::overload_cast<bliss::coarse_channel &>(&bliss::estimate_spectral_kurtosis),
          "fil_data"_a,
          "Compute spectral kurtosis of the given filterbank data");

    nb::enum_<bliss::noise_power_estimator>(m, "noise_power_estimator")
            .value("stddev", bliss::noise_power_estimator::STDDEV)
            .value("mad", bliss::noise_power_estimator::MEAN_ABSOLUTE_DEVIATION);

    nb::class_<bliss::noise_power_estimate_options>(m, "noise_power_estimate_options")
            .def(nb::init<>())
            .def_rw("estimator_method", &bliss::noise_power_estimate_options::estimator_method)
            .def_rw("masked_estimate", &bliss::noise_power_estimate_options::masked_estimate);

    m.def("estimate_noise_power", [](nb::ndarray<> arr, bliss::noise_power_estimate_options opts) {
        return bliss::estimate_noise_power(nb_to_bland(arr), opts);
    });
    m.def("estimate_noise_power",
          nb::overload_cast<bland::ndarray_deferred, bliss::noise_power_estimate_options>(&bliss::estimate_noise_power));
    m.def("estimate_noise_power",
          nb::overload_cast<bliss::coarse_channel, bliss::noise_power_estimate_options>(&bliss::estimate_noise_power));
    m.def("estimate_noise_power",
          nb::overload_cast<bliss::scan, bliss::noise_power_estimate_options>(&bliss::estimate_noise_power));
    m.def("estimate_noise_power",
          nb::overload_cast<bliss::observation_target, bliss::noise_power_estimate_options>(
                  &bliss::estimate_noise_power));
    m.def("estimate_noise_power",
          nb::overload_cast<bliss::cadence, bliss::noise_power_estimate_options>(&bliss::estimate_noise_power));

    nb::class_<bliss::noise_stats>(m, "noise_power")
            .def(nb::init<>())
            .def_prop_rw("noise_floor", &bliss::noise_stats::noise_floor, &bliss::noise_stats::set_noise_floor)
            .def_prop_ro("noise_amplitude", &bliss::noise_stats::noise_amplitude)
            .def_prop_rw("noise_power", &bliss::noise_stats::noise_power, &bliss::noise_stats::set_noise_power)
            .def("__repr__", &bliss::noise_stats::repr)
            // .def("__getstate__", [](bliss::noise_stats &self) {
            //       return std::make_tuple(self._noise_floor, self._noise_power);
            // })
            // .def("__setstate__", [](bliss::noise_stats &self, const std::tuple<float, float> &state_tuple) {
            //       new (&self) bliss::noise_stats;
            //       self._noise_floor = std::get<0>(state_tuple);
            //       self._noise_power = std::get<1>(state_tuple);
            // })
            ;
}
