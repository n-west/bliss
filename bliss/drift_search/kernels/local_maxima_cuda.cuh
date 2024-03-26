#pragma once

#include <drift_search/hit_search.hpp> // component

namespace bliss {

std::vector<component> find_local_maxima_above_threshold_cuda(coarse_channel        &dedrifted_coarse_channel,
                                                         float                  snr_threshold,
                                                         std::vector<bland::nd_coords> max_neighborhoods);

} // namespace bliss
