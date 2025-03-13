#pragma once

#include <bland/ndarray.hpp>

#include <drift_search/protohit_search.hpp> // component

#include <vector>

namespace bliss {


/**
 * Given noise stats do a combined threshold and cluster of nearby components
 */
std::vector<protohit>
find_components_above_threshold_per_drift_block_cuda(bland::ndarray                     doppler_spectrum,
                                      integrated_flags                 dedrifted_rfi,
                                      float                            noise_floor,
                                      std::vector<protohit_drift_info> noise_per_drift,
                                      float                            snr_threshold,
                                      int                              neighbor_l1_dist);

} // namespace bliss
