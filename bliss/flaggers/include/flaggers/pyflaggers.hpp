#pragma once

#include "magnitude.hpp"
#include "spectral_kurtosis.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

void bind_pyflaggers(nb::module_ m) {

    m.def("flag",
          &bliss::flag_spectral_kurtosis,
          "filterbank_data"_a,
          "return a masked copy of filterbank_data where spectral_kurtosis indicates non-gaussian samples");
}