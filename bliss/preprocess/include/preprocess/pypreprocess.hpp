#pragma once

#include "passband_static_equalize.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <fmt/format.h>

namespace nb = nanobind;
using namespace nb::literals;

void bind_pypreprocess(nb::module_ m) {

      m.def("firdes", bliss::firdes);

      m.def("gen_coarse_channel_inverse", bliss::gen_coarse_channel_inverse);

      m.def("equalize_passband_filter", bliss::equalize_passband_filter);

}