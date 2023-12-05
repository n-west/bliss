
#pragma once

#include "include/file_types/h5_filterbank_file.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;


void bind_pyfile_types(nb::module_ m) {


    nb::class_<bliss::h5_filterbank_file>(m, "h5_filterbank_file")
    .def(nb::init<std::string_view>())
    .def("__repr__", &bliss::h5_filterbank_file::repr)
    .def("data", &bliss::h5_filterbank_file::read_data)
    // .def("read_data_attr", &bliss::h5_filterbank_file::read_data_attr) // How do you bind this without knowing the type? :-|
    .def("mask", &bliss::h5_filterbank_file::read_mask)
    ;
    


}