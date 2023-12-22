

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include "hit_search.hpp"
#include "integrate_drifts.hpp"
#include "connected_components.hpp"
#include "local_maxima.hpp"

void bind_pydrift_search(nb::module_ m) {

    // Integration / dedoppler methods
    m.def("integrate_drifts", [](nb::ndarray<> spectrum, bliss::integrate_drifts_options options) {
        auto bland_spectrum  = nb_to_bland(spectrum);
        auto detection_plane = bliss::integrate_drifts(bland_spectrum, options);
        return detection_plane;
    });

    m.def("integrate_drifts",
          nb::overload_cast<bliss::filterbank_data, bliss::integrate_drifts_options>(&bliss::integrate_drifts));


    // General "component" class as intermediate between dedrifted clusters of prehits and hits
    nb::class_<bliss::component>(m, "component")
    .def_rw("locations", &bliss::component::locations)
    .def_rw("index_max", &bliss::component::index_max)
    .def_rw("max_integration", &bliss::component::max_integration);


    // hit definition (pre-capnproto serialized version)
    nb::class_<bliss::hit>(m, "hit")
    .def_rw("start_freq_index", &bliss::hit::start_freq_index)
    .def_rw("start_freq_MHz", &bliss::hit::start_freq_MHz)
    .def_rw("drift_rate_Hz_per_sec", &bliss::hit::drift_rate_Hz_per_sec)
    .def_rw("rate_index", &bliss::hit::rate_index)
    .def_rw("snr", &bliss::hit::snr)
    .def_rw("binwidth", &bliss::hit::binwidth)
    .def_rw("bandwidth", &bliss::hit::bandwidth);

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
          [](nb::ndarray<> threshold_mask, std::vector<bliss::nd_coords> neighborhood) { return bliss::find_components_in_binary_mask(nb_to_bland(threshold_mask), neighborhood); });

    // Local maxima method
    m.def("find_local_maxima_above_threshold", &bliss::find_local_maxima_above_threshold);

    nb::enum_<bliss::hit_search_methods>(m, "hit_search_methods")
    .value("connected_components", bliss::hit_search_methods::CONNECTED_COMPONENTS)
    .value("local_maxima", bliss::hit_search_methods::LOCAL_MAXIMA);

    nb::class_<bliss::hit_search_options>(m, "hit_search_options")
    .def(nb::init<>())
    .def_rw("method", &bliss::hit_search_options::method)
    .def_rw("snr_threshold", &bliss::hit_search_options::snr_threshold);
    
    // High-level "hit search" implementation
    m.def("hit_search", &bliss::hit_search);

}
