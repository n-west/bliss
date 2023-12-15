#pragma once

#include <core/doppler_spectrum.hpp>
#include <core/noise_power.hpp>

namespace bliss {

struct hit {
    int64_t start_freq_index;
    int64_t rate_index;
    float snr;
    int64_t bandwidth;
};

bland::ndarray hard_threshold_drifts(const bland::ndarray &dedrifted_spectrum,
                                     const noise_stats    &noise_stats,
                                     int64_t               integration_length,
                                     float                 snr_threshold);

using nd_coords = std::vector<int64_t>;

struct component {
    std::vector<nd_coords> locations;
    float                  s;
};

std::vector<component> find_components(const bland::ndarray &threshold_mask);

std::vector<hit>
hit_search(doppler_spectrum dedrifted_spectrum, noise_stats noise_stats, float snr_threshold);

} // namespace bliss