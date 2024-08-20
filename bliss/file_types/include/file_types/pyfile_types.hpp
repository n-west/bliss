
#pragma once

#include "events_file.hpp"
#include "h5_filterbank_file.hpp"
#include "cpnp_files.hpp"

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
            .def("read_data", &bliss::h5_filterbank_file::read_data)
            .def("read_mask", &bliss::h5_filterbank_file::read_mask)
            .def("get_file_path", &bliss::h5_filterbank_file::get_file_path);

    m.def("write_hits_to_capnp_file", &bliss::write_hits_to_capnp_file<std::list<bliss::hit>>);
    m.def("read_hits_from_capnp_file", &bliss::read_hits_from_capnp_file);
    m.def("write_scan_hits_to_file",
          &bliss::write_scan_hits_to_capnp_file,
          "scan_with_hits"_a,
          "file_path"_a,
          "write scan metadata and associated hits as cap'n proto messages to binary file at the given path");
    m.def("read_scan_hits_from_capnp_file",
          &bliss::read_scan_hits_from_capnp_file,
          "file_path"_a,
          "read cap'n proto serialized scan from file as written by `write_scan_hits_from_capnp_file`");
    m.def("write_observation_target_hits_to_capnp_files",
          &bliss::write_observation_target_hits_to_capnp_files,
          "observation_target"_a,
          "file_path"_a,
          "write an observation target's scan md and associated hits as cap'n proto messages to binary files matching "
          "the file_path the result will be one file per scan of the observation target");
    m.def("read_observation_target_hits_from_capnp_files", &bliss::read_observation_target_hits_from_capnp_files, "file_path"_a);
    m.def("write_cadence_hits_to_capnp_files",
          &bliss::write_cadence_hits_to_capnp_files,
          "cadence_with_hits"_a,
          "base_filename"_a,
          "write all detected hits for all scans of each observation target in a cadence as cap'n proto messages to "
          "binary files matching the file_path the result will be one file per scan for each observation target with "
          "filenames matching the pattern");
    m.def("read_cadence_hits_from_capnp_files",
          &bliss::read_cadence_hits_from_capnp_files,
          "base_filename"_a,
          "read cap'n proto serialized scan from file as written by `write_coarse_channel_hits_to_capnp_file`");
    m.def("write_events_to_file", &bliss::write_events_to_file);
    m.def("read_events_from_file", &bliss::read_events_from_file);
}