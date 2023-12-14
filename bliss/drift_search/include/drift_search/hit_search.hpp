#pragma once

#include <core/doppler_spectrum.hpp>
#include <core/noise_power.hpp>

namespace bliss {

// what does hitsearch return?
// we'll probably need a hit_search_options struct
void hit_search(doppler_spectrum dedrifted_spectrum, noise_power noise_stats, float snr_threshold);

} // namespace bliss