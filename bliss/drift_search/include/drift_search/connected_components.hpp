#pragma once

#include <core/protohit.hpp>

#include <drift_search/protohit_search.hpp>

#include <bland/ndarray.hpp>

#include <limits>
#include <vector>

namespace bliss {


/**
 * find clusters (components) of adjacent (in start frequency or drift rate) bins and group them together.
 *
 * Accepts a binary mask (1) of dtype uint8
 */
std::vector<protohit> find_components_in_binary_mask(const bland::ndarray &threshold_mask, std::vector<bland::nd_coords> neighborhood);

/**
 * Given noise stats do a combined threshold and cluster of nearby components
 */
std::vector<protohit>
find_components_above_threshold(bland::ndarray                           doppler_spectrum,
                                        integrated_flags                 dedrifted_rfi,
                                        float                            noise_floor,
                                        std::vector<protohit_drift_info> noise_per_drift,
                                        float                            snr_threshold,
                                        std::vector<bland::nd_coords>    neighborhood);

} // namespace bliss
