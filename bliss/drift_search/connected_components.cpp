
#include <drift_search/connected_components.hpp>
#include <drift_search/hit_search.hpp> // component

#include "kernels/connected_components_cpu.hpp"

using namespace bliss;

std::vector<protohit> bliss::find_components_in_binary_mask(const bland::ndarray         &mask,
                                                            std::vector<bland::nd_coords> neighborhood) {
    return find_components_in_binary_mask_cpu(mask, neighborhood);
}

std::vector<protohit>
bliss::find_components_above_threshold(bland::ndarray doppler_spectrum,
                                        integrated_flags                      dedrifted_rfi,
                                        float                                 noise_floor,
                                        std::vector<protohit_drift_info>      noise_per_drift,
                                        float                                 snr_threshold,
                                        std::vector<bland::nd_coords>         max_neighborhood) {
    return find_components_above_threshold_cpu(
            doppler_spectrum, dedrifted_rfi, noise_floor, noise_per_drift, snr_threshold, max_neighborhood);
}
