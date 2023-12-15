

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include "hit_search.hpp"
#include "integrate_drifts.hpp"

void bind_pydrift_search(nb::module_ m) {

    m.def("hard_threshold_drifts", &bliss::hard_threshold_drifts);
    m.def("hard_threshold_drifts",
          [](const nb::ndarray<>      &dedrifted_spectrum,
             const bliss::noise_stats &noise_stats,
             int64_t                   integration_length,
             float                     snr_threshold) {
              return bliss::hard_threshold_drifts(
                      nb_to_bland(dedrifted_spectrum), noise_stats, integration_length, snr_threshold);
          });

    nb::class_<bliss::component>(m, "component")
    .def_rw("locations", &bliss::component::locations)
    .def_rw("s", &bliss::component::s);

    m.def("find_components", &bliss::find_components);
    m.def("find_components",
          [](nb::ndarray<> threshold_mask) { return bliss::find_components(nb_to_bland(threshold_mask)); });

    m.def("integrate_drifts", [](nb::ndarray<> spectrum, bliss::integrate_drifts_options options) {
        auto bland_spectrum  = nb_to_bland(spectrum);
        auto detection_plane = bliss::integrate_drifts(bland_spectrum, options);
        return detection_plane;
    });

    m.def("integrate_drifts",
          nb::overload_cast<bliss::filterbank_data, bliss::integrate_drifts_options>(&bliss::integrate_drifts));

    // m.def("spectrum_sum", &bliss::spectrum_sum);
}
