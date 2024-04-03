#pragma once

#include <drift_search/protohit_search.hpp> // component

namespace bliss {

std::vector<protohit>
find_local_maxima_above_threshold_cuda(bland::ndarray                  doppler_spectrum,
                                      integrated_flags                 dedrifted_rfi,
                                      float                            noise_floor,
                                      std::vector<protohit_drift_info> noise_per_drift,
                                      float                            snr_threshold,
                                      std::vector<bland::nd_coords>    max_neighborhood);


} // namespace bliss
