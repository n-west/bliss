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

std::vector<hit>
hit_search(doppler_spectrum dedrifted_spectrum, noise_stats noise_stats, float snr_threshold);

} // namespace bliss