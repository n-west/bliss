
#pragma once

#include "h5_filterbank_file.hpp"
#include "hits_file.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;


void bind_pyfile_types(nb::module_ m) {

    nb::class_<bliss::h5_filterbank_file>(m, "h5_filterbank_file")
    .def(nb::init<std::string>())
    .def("__repr__", &bliss::h5_filterbank_file::repr)
    .def("data", &bliss::h5_filterbank_file::read_data)
    // .def("read_data_attr", &bliss::h5_filterbank_file::read_data_attr) // How do you bind this without knowing the type? :-|
    .def("mask", &bliss::h5_filterbank_file::read_mask)
    ;

    m.def("write_hits_to_file", &bliss::write_hits_to_file);
    m.def("read_hits_from_file", &bliss::read_hits_from_file);
    m.def("write_scan_hits_to_file", &bliss::write_scan_hits_to_file);
    m.def("read_scan_hits_from_file", &bliss::read_scan_hits_from_file);
    m.def("write_observation_target_hits_to_files", &bliss::write_observation_target_hits_to_files);
    m.def("read_observation_target_hits_from_files", &bliss::read_observation_target_hits_from_files);
    m.def("write_cadence_hits_to_files", &bliss::write_cadence_hits_to_files);
    m.def("read_cadence_hits_from_files", &bliss::read_cadence_hits_from_files);

}