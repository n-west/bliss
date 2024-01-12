#pragma once

#include "cadence.hpp"
#include "filterbank_data.hpp"
#include "scan.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

void bind_pycore(nb::module_ m) {

    nb::class_<bliss::filterbank_data>(m, "filterbank_data")
            .def(nb::init<std::string>())
            .def_prop_ro("data", &bliss::filterbank_data::data)
            .def_prop_ro("mask", nb::overload_cast<>(&bliss::filterbank_data::mask))
            // .def_prop_rw("mask", nb::overload_cast<>(&bliss::filterbank_data::mask), nb::overload_cast<const
            // bland::ndarray&>(&bliss::filterbank_data::mask))
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
            .def_prop_ro("za_start", &bliss::filterbank_data::za_start);

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

    nb::class_<bliss::integrate_drifts_options>(m, "integrate_drifts_options")
            .def(nb::init<>())
            // .def(nb::init<bool>(), )
            .def_rw("desmear", &bliss::integrate_drifts_options::desmear)
            .def_rw("low_rate", &bliss::integrate_drifts_options::low_rate)
            .def_rw("high_rate", &bliss::integrate_drifts_options::high_rate)
            .def_rw("rate_step_size", &bliss::integrate_drifts_options::rate_step_size);

    nb::class_<bliss::integrated_flags>(m, "integrated_flags")
            .def(nb::init<int64_t /*drifts*/, int64_t /*channels*/, bland::ndarray::dev /*device*/>())
            .def_rw("filter_rolloff", &bliss::integrated_flags::filter_rolloff)
            .def_rw("low_spectral_kurtosis", &bliss::integrated_flags::low_spectral_kurtosis)
            .def_rw("high_spectral_kurtosis", &bliss::integrated_flags::high_spectral_kurtosis)
            .def_rw("magnitude", &bliss::integrated_flags::magnitude)
            .def_rw("sigma_clip", &bliss::integrated_flags::sigma_clip);

    nb::class_<bliss::scan, bliss::filterbank_data>(m, "scan")
            .def(nb::init<bliss::filterbank_data /* fb_data */,
                          bland::ndarray /* doppler_spectrum */,
                          bliss::integrated_flags /* doppler_flags */,
                          bliss::integrate_drifts_options /* drift_parameters */>())
            .def(nb::init<const bliss::filterbank_data&>())
            .def("doppler_spectrum", nb::overload_cast<>(&bliss::scan::doppler_spectrum))
            .def("doppler_spectrum", nb::overload_cast<bland::ndarray>(&bliss::scan::doppler_spectrum))
            .def("doppler_flags", nb::overload_cast<>(&bliss::scan::doppler_flags))
            .def("doppler_flags", nb::overload_cast<bliss::integrated_flags>(&bliss::scan::doppler_flags))
            .def("drift_parameters", nb::overload_cast<>(&bliss::scan::dedoppler_options))
            .def("drift_parameters", nb::overload_cast<bliss::integrate_drifts_options>(&bliss::scan::dedoppler_options))
            .def_prop_rw("noise_estimate",
                         nb::overload_cast<>(&bliss::scan::noise_estimate),
                         nb::overload_cast<bliss::noise_stats>(&bliss::scan::noise_estimate));
    ;

    nb::class_<bliss::observation_target>(m, "observation_target")
            .def(nb::init<std::vector<bliss::filterbank_data>>())
            .def_rw("scans", &bliss::observation_target::_scans)
            .def_rw("target_name", &bliss::observation_target::_target_name);

    nb::class_<bliss::cadence>(m, "cadence")
            .def(nb::init<std::vector<bliss::observation_target>>())
            .def(nb::init<std::vector<std::vector<std::string_view>>>())
            .def_rw("observations", &bliss::cadence::_observations);
}