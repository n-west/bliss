#pragma once

#include <bland/ndarray.hpp>
#include "cadence.hpp"
#include "coarse_channel.hpp"
#include "scan.hpp"
#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>

namespace nb = nanobind;
using namespace nb::literals;

void bind_pycore(nb::module_ m) {

    nb::class_<bliss::frequency_drift_plane> pyfrequency_drift_plane(m, "frequency_drift_plane");
    pyfrequency_drift_plane
        .def("set_device", nb::overload_cast<std::string_view>(&bliss::frequency_drift_plane::set_device))
        .def("push_device", (&bliss::frequency_drift_plane::push_device))
        .def_prop_ro("integrated_drifts", &bliss::frequency_drift_plane::integrated_drift_plane)
        .def_prop_ro("integrated_rfi", &bliss::frequency_drift_plane::integrated_rfi)
        .def("drift_rate_info", &bliss::frequency_drift_plane::drift_rate_info);

    nb::class_<bliss::frequency_drift_plane::drift_rate>(pyfrequency_drift_plane, "drift_rate")
        .def_ro("desmeared_bins", &bliss::frequency_drift_plane::drift_rate::desmeared_bins)
        .def_ro("drift_rate_Hz_per_sec", &bliss::frequency_drift_plane::drift_rate::drift_rate_Hz_per_sec)
        .def_ro("drift_rate_slope", &bliss::frequency_drift_plane::drift_rate::drift_rate_slope)
        .def_ro("index_in_plane", &bliss::frequency_drift_plane::drift_rate::index_in_plane);

    nb::class_<bliss::coarse_channel>(m, "coarse_channel")
        .def_prop_ro("data", [](bliss::coarse_channel &self){return bland::ndarray(self.data());})
        .def_prop_ro("mask", [](bliss::coarse_channel &self){return bland::ndarray(self.mask());})
        .def_prop_ro("hits", &bliss::coarse_channel::hits)
        .def_prop_ro("az_start", &bliss::coarse_channel::az_start)
        .def_prop_ro("data_type", &bliss::coarse_channel::data_type)
        .def_prop_ro("fch1", &bliss::coarse_channel::fch1)
        .def_prop_ro("foff", &bliss::coarse_channel::foff)
        .def_prop_ro("machine_id", &bliss::coarse_channel::machine_id)
        .def_prop_ro("nbits", &bliss::coarse_channel::nbits)
        .def_prop_ro("nchans", &bliss::coarse_channel::nchans)
        .def_prop_ro("source_name", &bliss::coarse_channel::source_name)
        .def_prop_ro("src_dej", &bliss::coarse_channel::src_dej)
        .def_prop_ro("src_raj", &bliss::coarse_channel::src_raj)
        .def_prop_ro("telescope_id", &bliss::coarse_channel::telescope_id)
        .def_prop_ro("tsamp", &bliss::coarse_channel::tsamp)
        .def_prop_ro("tstart", &bliss::coarse_channel::tstart)
        .def_prop_ro("za_start", &bliss::coarse_channel::za_start)
        .def_prop_ro("noise_estimate", &bliss::coarse_channel::noise_estimate)
        .def("integrated_drift_plane", &bliss::coarse_channel::integrated_drift_plane)
        .def("device", &bliss::coarse_channel::device)
        .def("set_device", nb::overload_cast<bland::ndarray::dev&>(&bliss::coarse_channel::set_device))
        .def("set_device", nb::overload_cast<std::string_view>(&bliss::coarse_channel::set_device))
        .def("push_device", (&bliss::coarse_channel::push_device));


    nb::class_<bliss::scan>(m, "scan")
        .def(nb::init<std::string_view>(), "file_path"_a)
        .def(nb::init<std::string_view, int>(), "file_path"_a, "fine_channels_per_coarse"_a)
        .def("read_coarse_channel", &bliss::scan::read_coarse_channel)
        .def("peak_coarse_channel", &bliss::scan::peak_coarse_channel)
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
        .def("device", &bliss::scan::device)
        .def("set_device", nb::overload_cast<bland::ndarray::dev&>(&bliss::scan::set_device))
        .def("set_device", nb::overload_cast<std::string_view>(&bliss::scan::set_device))
        .def("push_device", (&bliss::scan::push_device));

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
        .def(nb::init<std::vector<bliss::scan>>(), "scans"_a)
        .def(nb::init<std::vector<std::string_view>>(), "scan_files"_a)
        .def(nb::init<std::vector<std::string_view>, int>(), "scan_files"_a, "fine_channels_per_coarse"_a)
        .def("validate_scan_consistency", &bliss::observation_target::validate_scan_consistency)
        .def("get_coarse_channel_with_frequency", &bliss::observation_target::get_coarse_channel_with_frequency, "frequency"_a)
        .def_prop_ro("number_coarse_channels", &bliss::observation_target::get_number_coarse_channels)
        .def("slice_observation_channels", &bliss::observation_target::slice_observation_channels, "start"_a=0, "count"_a=1)
        .def("device", &bliss::observation_target::device)
        .def("set_device", nb::overload_cast<bland::ndarray::dev&>(&bliss::observation_target::set_device))
        .def("set_device", nb::overload_cast<std::string_view>(&bliss::observation_target::set_device))
        .def_rw("scans", &bliss::observation_target::_scans)
        .def_rw("target_name", &bliss::observation_target::_target_name);

    nb::class_<bliss::cadence>(m, "cadence")
        .def(nb::init<std::vector<bliss::observation_target>>(), "observations"_a)
        .def(nb::init<std::vector<std::vector<std::string_view>>>(), "scan_files"_a)
        .def(nb::init<std::vector<std::vector<std::string_view>>, int>(), "scan_files"_a, "fine_channels_per_coarse"_a)
        .def("validate_scan_consistency", &bliss::cadence::validate_scan_consistency)
        .def("get_coarse_channel_with_frequency", &bliss::cadence::get_coarse_channel_with_frequency, "frequency"_a)
        .def_prop_ro("number_coarse_channels", &bliss::cadence::get_number_coarse_channels)
        .def("slice_cadence_channels", &bliss::cadence::slice_cadence_channels, "start"_a=0, "count"_a=1)
        .def_rw("observations", &bliss::cadence::_observations)
        .def("device", &bliss::cadence::device)
        .def("set_device", nb::overload_cast<bland::ndarray::dev&>(&bliss::cadence::set_device))
        .def("set_device", nb::overload_cast<std::string_view>(&bliss::cadence::set_device));
}
