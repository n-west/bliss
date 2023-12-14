

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "hit_search.hpp"
#include "integrate_drifts.hpp"

void bind_pydrift_search(nb::module_ m) {

    m.def("integrate_drifts", [](nb::ndarray<> spectrum, bliss::integrate_drifts_options options) {
        auto bland_spectrum = nb_to_bland(spectrum);
        auto detection_plane = bliss::integrate_drifts(bland_spectrum, options);
        return detection_plane;
    });

    m.def("integrate_drifts", nb::overload_cast<bliss::filterbank_data, bliss::integrate_drifts_options>(&bliss::integrate_drifts));

    // m.def("spectrum_sum", &bliss::spectrum_sum);
}
