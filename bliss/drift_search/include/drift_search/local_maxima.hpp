#pragma once

#include "hit_search.hpp"

namespace bliss {

    std::vector<component> find_local_maxima_above_threshold(scan &dedrifted_spectrum, float snr_threshold, std::vector<nd_coords> max_neighborhoods);

} // namespace bliss
