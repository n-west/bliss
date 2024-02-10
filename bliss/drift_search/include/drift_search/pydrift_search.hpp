

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "connected_components.hpp"
#include "event_search.hpp"
#include "hit_search.hpp"
#include "integrate_drifts.hpp"
#include "local_maxima.hpp"

void bind_pydrift_search(nb::module_ m) {

    // Integration / dedoppler methods
    m.def("integrate_drifts", [](nb::ndarray<> spectrum, bliss::integrate_drifts_options options) {
        auto bland_spectrum  = nb_to_bland(spectrum);
        auto detection_plane = bliss::integrate_drifts(bland_spectrum, options);
        return detection_plane;
    });

    m.def("integrate_drifts",
          nb::overload_cast<bliss::scan, bliss::integrate_drifts_options>(&bliss::integrate_drifts));

    m.def("integrate_drifts",
          nb::overload_cast<bliss::observation_target, bliss::integrate_drifts_options>(&bliss::integrate_drifts));

    m.def("integrate_drifts",
          nb::overload_cast<bliss::cadence, bliss::integrate_drifts_options>(&bliss::integrate_drifts));

    // General "component" class as intermediate between dedrifted clusters of prehits and hits
    nb::class_<bliss::component>(m, "component")
            .def_rw("locations", &bliss::component::locations)
            .def_rw("index_max", &bliss::component::index_max)
            .def_rw("max_integration", &bliss::component::max_integration)
            .def_rw("rfi_counts", &bliss::component::rfi_counts);

    // hit definition (pre-capnproto serialized version)
    nb::class_<bliss::hit>(m, "hit")
            .def_rw("start_freq_index", &bliss::hit::start_freq_index)
            .def_rw("start_freq_MHz", &bliss::hit::start_freq_MHz)
            .def_rw("drift_rate_Hz_per_sec", &bliss::hit::drift_rate_Hz_per_sec)
            .def_rw("rate_index", &bliss::hit::rate_index)
            .def_rw("snr", &bliss::hit::snr)
            .def_rw("rfi_counts", &bliss::hit::rfi_counts)
            .def_rw("binwidth", &bliss::hit::binwidth)
            .def_rw("bandwidth", &bliss::hit::bandwidth)
            .def("__repr__", &bliss::hit::repr)
            .def("__getstate__", &bliss::hit::get_state)
            .def("__setstate__", [](bliss::hit &self, const bliss::hit::state_tuple &state) {
                new (&self) bliss::hit;
                self.set_state(state);
            });

    // Generic thresholding methods
    m.def("hard_threshold_drifts", &bliss::hard_threshold_drifts);
    m.def("hard_threshold_drifts",
          [](const nb::ndarray<>      &dedrifted_spectrum,
             const bliss::noise_stats &noise_stats,
             int64_t                   integration_length,
             float                     snr_threshold) {
              return bliss::hard_threshold_drifts(
                      nb_to_bland(dedrifted_spectrum), noise_stats, integration_length, snr_threshold);
          });

    // ** hit finding methods **

    // Connected components variations
    m.def("find_components_above_threshold", &bliss::find_components_above_threshold);

    m.def("find_components_in_binary_mask", &bliss::find_components_in_binary_mask);
    m.def("find_components_in_binary_mask",
          [](nb::ndarray<> threshold_mask, std::vector<bliss::nd_coords> neighborhood) {
              return bliss::find_components_in_binary_mask(nb_to_bland(threshold_mask), neighborhood);
          });

    // Local maxima method
    m.def("find_local_maxima_above_threshold", &bliss::find_local_maxima_above_threshold);

    nb::enum_<bliss::hit_search_methods>(m, "hit_search_methods")
            .value("connected_components", bliss::hit_search_methods::CONNECTED_COMPONENTS)
            .value("local_maxima", bliss::hit_search_methods::LOCAL_MAXIMA);

    nb::class_<bliss::hit_search_options>(m, "hit_search_options")
            .def(nb::init<>())
            .def_rw("method", &bliss::hit_search_options::method)
            .def_rw("snr_threshold", &bliss::hit_search_options::snr_threshold)
            .def_rw("neighborhood", &bliss::hit_search_options::neighborhood);

    // High-level "hit search" implementation
    m.def("hit_search", nb::overload_cast<bliss::scan, bliss::hit_search_options>(&bliss::hit_search));
    m.def("hit_search", nb::overload_cast<bliss::observation_target, bliss::hit_search_options>(&bliss::hit_search));
    m.def("hit_search", nb::overload_cast<bliss::cadence, bliss::hit_search_options>(&bliss::hit_search));

    nb::class_<bliss::event>(m, "event")
            .def_rw("hits", &bliss::event::hits)
            .def_rw("average_drift_rate_Hz_per_sec", &bliss::event::average_drift_rate_Hz_per_sec)
            .def_rw("average_power", &bliss::event::average_power)
            .def_rw("average_snr", &bliss::event::average_snr)
            .def_rw("event_start_seconds", &bliss::event::event_start_seconds)
            .def_rw("event_end_seconds", &bliss::event::event_end_seconds)
            .def_rw("starting_frequency_Hz", &bliss::event::starting_frequency_Hz)
            .def("__repr__", &bliss::event::repr)
            .def("__getstate__", [](const bliss::event &self) { return self.hits; })
            //     .def("__setstate__", [](bliss::event &self, const std::vector<bliss::hit> &state) {
            //         new (&self) bliss::event{.hits = state};
            //     })
            ;

    m.def("event_search", &bliss::event_search);
}
