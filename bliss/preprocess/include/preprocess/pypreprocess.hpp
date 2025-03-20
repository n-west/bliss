#pragma once

#include "excise_dc.hpp"
#include "normalize.hpp"
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

      m.def("equalize_passband_filter", nb::overload_cast<bliss::coarse_channel, bland::ndarray, bool>(bliss::equalize_passband_filter), "coarse_channel"_a, "response"_a, "validate"_a=false);
      m.def("equalize_passband_filter", nb::overload_cast<bliss::coarse_channel, std::string_view, bland::ndarray::datatype, bool>(bliss::equalize_passband_filter), "coarse_channel"_a, "response_path"_a, "response_dtype"_a=bland::ndarray::datatype("float"), "validate"_a=false);

      m.def("equalize_passband_filter", nb::overload_cast<bliss::scan, bland::ndarray, bool>(bliss::equalize_passband_filter), "scan"_a, "response"_a, "validate"_a=false);
      m.def("equalize_passband_filter", nb::overload_cast<bliss::scan, std::string_view, bland::ndarray::datatype, bool>(bliss::equalize_passband_filter), "scan"_a, "response_path"_a, "response_dtype"_a=bland::ndarray::datatype("float"), "validate"_a=false);

      m.def("equalize_passband_filter", nb::overload_cast<bliss::observation_target, bland::ndarray, bool>(bliss::equalize_passband_filter), "observation_target"_a, "response"_a, "validate"_a=false);
      m.def("equalize_passband_filter", nb::overload_cast<bliss::observation_target, std::string_view, bland::ndarray::datatype, bool>(bliss::equalize_passband_filter), "observation_target"_a, "response_path"_a, "response_dtype"_a=bland::ndarray::datatype("float"), "validate"_a=false);

      m.def("equalize_passband_filter", nb::overload_cast<bliss::cadence, bland::ndarray, bool>(bliss::equalize_passband_filter), "cadence"_a, "response"_a, "validate"_a=false);
      m.def("equalize_passband_filter", nb::overload_cast<bliss::cadence, std::string_view, bland::ndarray::datatype, bool>(bliss::equalize_passband_filter), "cadence"_a, "response_path"_a, "response_dtype"_a=bland::ndarray::datatype("float"), "validate"_a=false);

      m.def("normalize", nb::overload_cast<bliss::coarse_channel>(bliss::normalize));
      m.def("normalize", nb::overload_cast<bliss::scan>(bliss::normalize));
      m.def("normalize", nb::overload_cast<bliss::observation_target>(bliss::normalize));
      m.def("normalize", nb::overload_cast<bliss::cadence>(bliss::normalize));


      m.def("excise_dc", nb::overload_cast<bliss::coarse_channel>(bliss::excise_dc));
      m.def("excise_dc", nb::overload_cast<bliss::scan>(bliss::excise_dc));
      m.def("excise_dc", nb::overload_cast<bliss::observation_target>(bliss::excise_dc));
      m.def("excise_dc", nb::overload_cast<bliss::cadence>(bliss::excise_dc));

}