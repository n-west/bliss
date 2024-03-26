#pragma once

#include "protohit_search.hpp"

namespace bliss {

std::vector<protohit> find_local_maxima_above_threshold(bland::ndarray   doppler_spectrum,
                                    integrated_flags                     dedrifted_rfi,
                                    std::vector<std::pair<float, float>> noise_and_thresholds_per_drift,
                                    std::vector<bland::nd_coords>        max_neighborhood);

} // namespace bliss
