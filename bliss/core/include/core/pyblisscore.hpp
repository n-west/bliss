#pragma once

#include "cadence.hpp"
#include "coarse_channel.hpp"
#include "scan.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

namespace nb = nanobind;
using namespace nb::literals;

void bind_pycore(nb::module_ m) {

    nb::class_<bliss::frequency_drift_plane>(m, "frequency_drift_plane")
        .def_ro("integrated_drifts", &bliss::frequency_drift_plane::_integrated_drifts)
    ;

    nb::class_<bliss::coarse_channel>(m, "coarse_channel")
            .def_ro("data", &bliss::coarse_channel::_data)
            .def_ro("mask", &bliss::coarse_channel::_mask)
            .def_ro("hits", &bliss::coarse_channel::_hits)
            .def_ro("az_start", &bliss::coarse_channel::_az_start)
            .def_ro("data_type", &bliss::coarse_channel::_data_type)
            .def_ro("fch1", &bliss::coarse_channel::_fch1)
            .def_ro("foff", &bliss::coarse_channel::_foff)
            .def_ro("machine_id", &bliss::coarse_channel::_machine_id)
            .def_ro("nbits", &bliss::coarse_channel::_nbits)
            .def_ro("nchans", &bliss::coarse_channel::_nchans)
            .def_ro("source_name", &bliss::coarse_channel::_source_name)
            .def_ro("src_dej", &bliss::coarse_channel::_src_dej)
            .def_ro("src_raj", &bliss::coarse_channel::_src_raj)
            .def_ro("telescope_id", &bliss::coarse_channel::_telescope_id)
            .def_ro("tsamp", &bliss::coarse_channel::_tsamp)
            .def_ro("tstart", &bliss::coarse_channel::_tstart)
            .def_ro("za_start", &bliss::coarse_channel::_za_start)
            .def_ro("noise_estimate", &bliss::coarse_channel::_noise_stats)
            .def("integrated_drift_plane", &bliss::coarse_channel::integrated_drift_plane)
                ;

    nb::class_<bliss::scan>(m, "scan")
            .def(nb::init<std::string_view>(), "file_path"_a)
            .def("get_coarse_channel", &bliss::scan::get_coarse_channel)
            .def("get_channelization", &bliss::scan::get_channelization)
            .def("get_coarse_channel_with_frequency", &bliss::scan::get_coarse_channel_with_frequency, "frequency"_a)
            .def("slice_scan_channels", &bliss::scan::slice_scan_channels, "start"_a=0, "count"_a=1)
            .def("hits", &bliss::scan::hits)
            .def_prop_ro("num_coarse_channels", &bliss::scan::get_number_coarse_channels)
            .def_prop_ro("az_start", &bliss::scan::az_start)
            .def_prop_ro("data_type",&bliss::scan::data_type)
            .def_prop_ro("fch1", &bliss::scan::fch1)
            .def_prop_ro("foff", &bliss::scan::foff)
            .def_prop_ro("machine_id", &bliss::scan::machine_id)
            .def_prop_ro("nbits", &bliss::scan::nbits)
            .def_prop_ro("nchans", &bliss::scan::nchans)
            .def_prop_ro("source_name", &bliss::scan::source_name)
            .def_prop_ro("src_dej", &bliss::scan::src_dej)
            .def_prop_ro("src_raj", &bliss::scan::src_raj)
            .def_prop_ro("telescope_id", &bliss::scan::telescope_id)
            .def_prop_ro("tsamp", &bliss::scan::tsamp)
            .def_prop_ro("tstart", &bliss::scan::tstart)
            .def_prop_ro("za_start", &bliss::scan::za_start)
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

    nb::class_<bliss::observation_target>(m, "observation_target")
            .def(nb::init<std::vector<bliss::scan>>())
            .def(nb::init<std::vector<std::string_view>>())
            .def_rw("scans", &bliss::observation_target::_scans)
            .def("slice_observation_channels", &bliss::observation_target::slice_observation_channels, "start"_a=0, "count"_a=1)
            .def_rw("target_name", &bliss::observation_target::_target_name);

    nb::class_<bliss::cadence>(m, "cadence")
            .def(nb::init<std::vector<bliss::observation_target>>())
            .def(nb::init<std::vector<std::vector<std::string_view>>>())
            .def("slice_cadence_channels", &bliss::cadence::slice_cadence_channels, "start"_a=0, "count"_a=1)
            .def_rw("observations", &bliss::cadence::_observations);
}