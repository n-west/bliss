#pragma once

#include "filter_rolloff.hpp"
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
          &bliss::flag_spectral_kurtosis,
          "filterbank_data"_a,
          "lower_threshold"_a,
          "upper_threshold"_a,
          "return a masked copy of filterbank_data where estimate_spectral_kurtosis indicates non-gaussian samples");

    m.def("flag_filter_rolloff",
          &bliss::flag_filter_rolloff,
          "filterbank_data"_a,
          "rolloff_width"_a,
          "return a masked copy of filterbank_data where the frequency edges are flagged according to rolloff width");

    m.def("flag_magnitude",
          &bliss::flag_magnitude,
          "filterbank_data"_a,
          "sigma"_a,
          "return a masked copy of filterbank_data where magnitude exceeds the mean by given sigma");
}