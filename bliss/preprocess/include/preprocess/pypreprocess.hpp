#pragma once

#include "passband_static_equalize.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <fmt/format.h>

namespace nb = nanobind;
using namespace nb::literals;

void bind_pypreprocess(nb::module_ m) {

      m.def("firdes", bliss::firdes);

      m.def("gen_coarse_channel_response",
            bliss::gen_coarse_channel_response,
            "fine_per_coarse"_a,
            "num_coarse_channels"_a,
            "taps_per_channel"_a,
            "window"_a     = "hamming",
            "device_str"_a = "cpu");

      m.def("equalize_passband_filter", nb::overload_cast<bliss::coarse_channel, bland::ndarray>(bliss::equalize_passband_filter));
      m.def("equalize_passband_filter", nb::overload_cast<bliss::coarse_channel, std::string_view, bland::ndarray::datatype>(bliss::equalize_passband_filter));

}