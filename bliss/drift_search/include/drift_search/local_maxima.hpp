#pragma once

#include <core/protohit.hpp>

#include <core/frequency_drift_plane.hpp>

#include <bland/ndarray.hpp>
#include <bland/stride_helper.hpp> // bland::nd_coord

namespace bliss {

std::vector<protohit> find_local_maxima_above_threshold(bland::ndarray   doppler_spectrum,
                                        integrated_flags                 dedrifted_rfi,
                                        float                            noise_floor,
                                        std::vector<protohit_drift_info> noise_per_drift,
                                        float                            snr_threshold,
                                        std::vector<bland::nd_coords>    neighborhood);

} // namespace bliss
