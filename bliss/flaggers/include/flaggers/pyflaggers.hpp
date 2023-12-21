#pragma once

#include "filter_rolloff.hpp"
#include "flag_values.hpp"
#include "magnitude.hpp"
#include "spectral_kurtosis.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

void bind_pyflaggers(nb::module_ m) {

    m.def("flag_spectral_kurtosis",
          nb::overload_cast<bliss::filterbank_data, float, float>(&bliss::flag_spectral_kurtosis),
          "filterbank_data"_a,
          "lower_threshold"_a,
          "upper_threshold"_a,
          "return a masked copy of filterbank_data where estimate_spectral_kurtosis indicates non-gaussian samples");

    m.def("flag_spectral_kurtosis",
          nb::overload_cast<bliss::observation_target, float, float>(&bliss::flag_spectral_kurtosis),
          "filterbank_data"_a,
          "lower_threshold"_a,
          "upper_threshold"_a,
          "return an observation target where all filterbank_data have non-gaussian samples flagged by spectral "
          "kurtosis");

    m.def("flag_spectral_kurtosis",
          nb::overload_cast<bliss::cadence, float, float>(&bliss::flag_spectral_kurtosis),
          "filterbank_data"_a,
          "lower_threshold"_a,
          "upper_threshold"_a,
          "return a cadence with non-gaussian samples are flagged");

    m.def("flag_filter_rolloff",
          nb::overload_cast<bliss::filterbank_data, float>(&bliss::flag_filter_rolloff),
          "filterbank_data"_a,
          "rolloff_width"_a,
          "return a masked copy of filterbank_data where the frequency edges are flagged according to rolloff width");

    m.def("flag_filter_rolloff",
          nb::overload_cast<bliss::observation_target, float>(&bliss::flag_filter_rolloff),
          "filterbank_data"_a,
          "rolloff_width"_a,
          "return a masked copy of filterbank_data where the frequency edges are flagged according to rolloff width");

    m.def("flag_filter_rolloff",
          nb::overload_cast<bliss::cadence, float>(&bliss::flag_filter_rolloff),
          "filterbank_data"_a,
          "rolloff_width"_a,
          "return a masked copy of filterbank_data where the frequency edges are flagged according to rolloff width");

    m.def("flag_magnitude",
          nb::overload_cast<const bland::ndarray&, float>(&bliss::flag_magnitude),
          "filterbank_data"_a,
          "threshold"_a,
          "return a masked copy of filterbank_data where magnitude exceeds the mean by given sigma");

    m.def("flag_magnitude",
          nb::overload_cast<bliss::filterbank_data, float>(&bliss::flag_magnitude),
          "filterbank_data"_a,
          "threshold"_a,
          "return a masked copy of filterbank_data where magnitude exceeds the mean by given sigma");

    m.def("flag_magnitude",
          nb::overload_cast<bliss::filterbank_data>(&bliss::flag_magnitude),
          "filterbank_data"_a,
          "return a masked copy of filterbank_data where magnitude exceeds the mean by given sigma");

    m.def("flag_magnitude",
          nb::overload_cast<bliss::observation_target>(&bliss::flag_magnitude),
          "observation"_a,
          "return a masked copy of filterbank_data where magnitude exceeds the mean by given sigma");

    m.def("flag_magnitude",
          nb::overload_cast<bliss::cadence>(&bliss::flag_magnitude),
          "cadence"_a,
          "return a masked copy of cadence where magnitude exceeds the mean by given sigma");

    nb::enum_<bliss::flag_values>(m, "flag_values")
            .value("unflagged", bliss::flag_values::unflagged)
            .value("filter_rolloff", bliss::flag_values::filter_rolloff)
            .value("low_spectral_kurtosis", bliss::flag_values::low_spectral_kurtosis)
            .value("high_spectral_kurtosis", bliss::flag_values::high_spectral_kurtosis)
            .value("magnitude", bliss::flag_values::magnitude)
            .value("sigma_clip", bliss::flag_values::sigma_clip);
}