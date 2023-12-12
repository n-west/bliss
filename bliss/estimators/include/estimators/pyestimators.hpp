#pragma once

#include "noise_estimate.hpp"
#include "spectral_kurtosis.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

void bind_pyestimators(nb::module_ m) {

    m.def("spectral_kurtosis", &bliss::spectral_kurtosis, "spectrum_grid"_a, "N"_a, "M"_a, "d"_a = 1,
        "Compute spectral kurtosis of the given spectra"
    );


}
