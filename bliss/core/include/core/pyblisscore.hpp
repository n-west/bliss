#pragma once

#include "filterbank_data.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

void bind_pycore(nb::module_ m) {

    nb::class_<bliss::filterbank_data>(m, "filterbank_data")
    .def(nb::init<std::string>())
    .def_prop_ro("data", &bliss::filterbank_data::data)
    .def_prop_ro("mask", &bliss::filterbank_data::mask)
    .def_prop_ro("az_start", &bliss::filterbank_data::az_start)
    .def_prop_ro("data_type", &bliss::filterbank_data::data_type)
    .def_prop_ro("fch1", &bliss::filterbank_data::fch1)
    .def_prop_ro("foff", &bliss::filterbank_data::foff)
    .def_prop_ro("machine_id", &bliss::filterbank_data::machine_id)
    .def_prop_ro("nbits", &bliss::filterbank_data::nbits)
    .def_prop_ro("nchans", &bliss::filterbank_data::nchans)
    .def_prop_ro("source_name", &bliss::filterbank_data::source_name)
    .def_prop_ro("src_dej", &bliss::filterbank_data::src_dej)
    .def_prop_ro("src_raj", &bliss::filterbank_data::src_raj)
    .def_prop_ro("telescope_id", &bliss::filterbank_data::telescope_id)
    .def_prop_ro("tsamp", &bliss::filterbank_data::tsamp)
    .def_prop_ro("tstart", &bliss::filterbank_data::tstart)
    .def_prop_ro("za_start", &bliss::filterbank_data::za_start)
    ;


    //  * *DIMENSION_LABELS
    //  * *az_start
    //  * *data_type
    //  * *fch1
    //  * *foff
    //  * *machine_id
    //  * *nbits
    //  * *nchans
    //  * *nifs
    //  * *source_name
    //  * *src_dej
    //  * *src_raj
    //  * *telescope_id
    //  * *tsamp
    //  * *tstart
    //  * *za_start

}