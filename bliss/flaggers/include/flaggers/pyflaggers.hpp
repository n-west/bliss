#pragma once

#include "filter_rolloff.hpp"
#include "magnitude.hpp"
#include "sigmaclip.hpp"
#include "spectral_kurtosis.hpp"

#include <core/flag_values.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <iostream>

namespace nb = nanobind;
using namespace nb::literals;

void bind_pyflaggers(nb::module_ m) {

    // Spectral Kurtosis
    m.def("flag_spectral_kurtosis",
          nb::overload_cast<bliss::coarse_channel, float, float, float>(&bliss::flag_spectral_kurtosis),
          "cc_data"_a,
          "lower_threshold"_a,
          "upper_threshold"_a,
          "d"_a = 2.0,
          "return a masked copy of scan where estimate_spectral_kurtosis indicates non-gaussian samples");

    m.def("flag_spectral_kurtosis",
          nb::overload_cast<bliss::scan, float, float, float>(&bliss::flag_spectral_kurtosis),
          "scan"_a,
          "lower_threshold"_a,
          "upper_threshold"_a,
          "d"_a = 2.0,
          "return a masked copy of scan where estimate_spectral_kurtosis indicates non-gaussian samples");

    m.def("flag_spectral_kurtosis",
          nb::overload_cast<bliss::observation_target, float, float, float>(&bliss::flag_spectral_kurtosis),
          "observation"_a,
          "lower_threshold"_a,
          "upper_threshold"_a,
          "d"_a = 2.0,
          "return an observation target where all scan have non-gaussian samples flagged by spectral "
          "kurtosis");

    m.def("flag_spectral_kurtosis",
          nb::overload_cast<bliss::cadence, float, float, float>(&bliss::flag_spectral_kurtosis),
          "cadence"_a,
          "lower_threshold"_a,
          "upper_threshold"_a,
          "d"_a=2.0,
          "return a cadence with non-gaussian samples are flagged");

    // Rolloff
    m.def("flag_filter_rolloff",
          nb::overload_cast<bliss::coarse_channel, float>(&bliss::flag_filter_rolloff),
          "cc_data"_a,
          "rolloff_width"_a,
          "return a masked copy of scan where the frequency edges are flagged according to rolloff width");

    m.def("flag_filter_rolloff",
          nb::overload_cast<bliss::scan, float>(&bliss::flag_filter_rolloff),
          "scan"_a,
          "rolloff_width"_a,
          "return a masked copy of scan where the frequency edges are flagged according to rolloff width");

    m.def("flag_filter_rolloff",
          nb::overload_cast<bliss::observation_target, float>(&bliss::flag_filter_rolloff),
          "observation"_a,
          "rolloff_width"_a,
          "return a masked copy of scan where the frequency edges are flagged according to rolloff width");

    m.def("flag_filter_rolloff",
          nb::overload_cast<bliss::cadence, float>(&bliss::flag_filter_rolloff),
          "cadence"_a,
          "rolloff_width"_a,
          "return a masked copy of scan where the frequency edges are flagged according to rolloff width");

    // Magnitude
    m.def("flag_magnitude",
          nb::overload_cast<const bland::ndarray &, float>(&bliss::flag_magnitude),
          "scan"_a,
          "threshold"_a,
          "return a masked copy of scan where magnitude exceeds the mean by given sigma");

    m.def("flag_magnitude",
          nb::overload_cast<bliss::coarse_channel>(&bliss::flag_magnitude),
          "cc_data"_a,
          "return a masked copy of coarse_channel where magnitude exceeds the mean by given sigma");

    m.def("flag_magnitude",
          nb::overload_cast<bliss::coarse_channel, float>(&bliss::flag_magnitude),
          "cc_data"_a,
          "threshold"_a,
          "return a masked copy of coarse_channel where magnitude exceeds the mean by given sigma");

    m.def("flag_magnitude",
          nb::overload_cast<bliss::scan>(&bliss::flag_magnitude),
          "scan"_a,
          "return a masked copy of scan where magnitude exceeds the mean by given sigma");

    m.def("flag_magnitude",
          nb::overload_cast<bliss::observation_target>(&bliss::flag_magnitude),
          "observation"_a,
          "return a masked copy of scan where magnitude exceeds the mean by given sigma");

    m.def("flag_magnitude",
          nb::overload_cast<bliss::cadence>(&bliss::flag_magnitude),
          "cadence"_a,
          "return a masked copy of cadence where magnitude exceeds the mean by given sigma");

    // Sigma clipped
// bland::ndarray bliss::flag_sigmaclip(const bland::ndarray &data, int max_iter, float low, float high) {
    m.def("flag_sigmaclip",
          nb::overload_cast<const bland::ndarray&, int, float, float>(&bliss::flag_sigmaclip),
          "x"_a,
          "max_iter"_a=3,
          "lower_threshold"_a=4.0f,
          "upper_threshold"_a=4.0f,
          "return a mask indicating which values are above/below the sigma clipped threshold");

    m.def("flag_sigmaclip",
          nb::overload_cast<bliss::coarse_channel, int, float, float>(&bliss::flag_sigmaclip),
          "cc_data"_a,
          "max_iter"_a=3,
          "lower_threshold"_a=4.0f,
          "upper_threshold"_a=4.0f,
          "return a masked copy of coarse_channel where sigmaclip indicates sample values outside of the deviation range given");

    m.def("flag_sigmaclip",
          nb::overload_cast<bliss::scan, int, float, float>(&bliss::flag_sigmaclip),
          "scan"_a,
          "max_iter"_a=3,
          "lower_threshold"_a=4.0f,
          "upper_threshold"_a=4.0f,
          "return a masked copy of scan where sigmaclip indicates sample values outside of the deviation range given");

    m.def("flag_sigmaclip",
          nb::overload_cast<bliss::observation_target, int, float, float>(&bliss::flag_sigmaclip),
          "observation"_a,
          "max_iter"_a=3,
          "lower_threshold"_a=4.0f,
          "upper_threshold"_a=4.0f,
          "return a masked copy of observation target where sigmaclip indicates sample values outside of the deviation range given");

    m.def("flag_sigmaclip",
          nb::overload_cast<bliss::cadence, int, float, float>(&bliss::flag_sigmaclip),
          "cadence"_a,
          "max_iter"_a=3,
          "lower_threshold"_a=4.0f,
          "upper_threshold"_a=4.0f,
          "return a masked copy of cadence where sigmaclip indicates sample values outside of the deviation range given");

    static auto flag_names = std::vector<std::string>{"unflagged",
                                                      "filter_rolloff",
                                                      "low_spectral_kurtosis",
                                                      "high_spectral_kurtosis",
                                                      "RESERVED_0",
                                                      "magnitude",
                                                      "sigmaclip",
                                                      "RESERVED_1",
                                                      "RESERVED_2"};

    nb::enum_<bliss::flag_values>(m, "flag_values")
            .value("unflagged", bliss::flag_values::unflagged)
            .value("filter_rolloff", bliss::flag_values::filter_rolloff)
            .value("low_spectral_kurtosis", bliss::flag_values::low_spectral_kurtosis)
            .value("high_spectral_kurtosis", bliss::flag_values::high_spectral_kurtosis)
            .value("magnitude", bliss::flag_values::magnitude)
            .value("sigma_clip", bliss::flag_values::sigma_clip)
            .def("__getstate__",
                 [](const bliss::flag_values &self) {
                     std::cout << "get state is " << static_cast<int>(self) << std::endl;
                     return std::make_tuple(static_cast<uint8_t>(self));
                 })
            // .def("__setstate__", [](bliss::flag_values &self, const std::tuple<uint8_t> &state) {
            //       std::cout << "called setstate" << std::endl;
            //       self = static_cast<bliss::flag_values>(std::get<0>(state));
            // })

            // .def("__getstate__", [](const bliss::flag_values &self) { return std::string("filter_rolloff"); })
            .def("__setstate__", [](bliss::flag_values &self, const std::tuple<uint8_t> &state) {
                std::cout << "called setstate" << std::endl;
                self = static_cast<bliss::flag_values>(std::get<0>(state));
            });
}