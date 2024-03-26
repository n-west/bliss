#pragma once

#include <drift_search/protohit_search.hpp> // component

namespace bliss {

std::vector<protohit>
find_local_maxima_above_threshold_cuda(bland::ndarray                       doppler_spectrum,
                                      integrated_flags                     dedrifted_rfi,
                                      std::vector<std::pair<float, float>> noise_and_thresholds_per_drift,
                                      std::vector<bland::nd_coords>        max_neighborhood);


} // namespace bliss
