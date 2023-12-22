#pragma once

#include "hit_search.hpp"
#include <bland/ndarray.hpp>
#include <core/doppler_spectrum.hpp>
#include <core/noise_power.hpp>
#include <limits>
#include <vector>

namespace bliss {


/**
 * find clusters (components) of adjacent (in start frequency or drift rate) bins and group them together.
 *
 * Accepts a binary mask (1) of dtype uint8
 */
std::vector<component> find_components_in_binary_mask(const bland::ndarray &threshold_mask, std::vector<nd_coords> neighborhood);

/**
 * Given noise stats do a combined threshold and cluster of nearby components
 */
std::vector<component>
find_components_above_threshold(doppler_spectrum &dedrifted_spectrum, noise_stats noise_stats, float snr_threshold, std::vector<nd_coords> neighborhood);

} // namespace bliss
