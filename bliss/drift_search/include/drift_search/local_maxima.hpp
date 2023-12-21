#pragma once

#include "hit_search.hpp"

namespace bliss {

    std::vector<component> find_local_maxima_above_threshold(doppler_spectrum &dedrifted_spectrum, noise_stats noise_stats, float snr_threshold);

} // namespace bliss
