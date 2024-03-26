
#include <drift_search/connected_components.hpp>
#include <drift_search/hit_search.hpp> // component

#include "kernels/connected_components_cpu.hpp"

using namespace bliss;


std::vector<component> bliss::find_components_in_binary_mask(const bland::ndarray  &mask,
                                                             std::vector<bland::nd_coords> neighborhood) {
    return find_components_in_binary_mask_cpu(mask, neighborhood);
}

std::vector<component> bliss::find_components_above_threshold(coarse_channel        &dedrifted_coarse_channel,
                                                              float                  snr_threshold,
                                                              std::vector<bland::nd_coords> neighborhood) {
    auto noise_stats = dedrifted_coarse_channel.noise_estimate();

    auto drift_plane = dedrifted_coarse_channel.integrated_drift_plane();
    auto integration_length = drift_plane.integration_steps();

    const auto noise_and_thresholds_per_drift = compute_noise_and_snr_thresholds(noise_stats, integration_length, drift_plane.drift_rate_info(), snr_threshold);


    auto doppler_spectrum = drift_plane.integrated_drift_plane();
    auto dedrifted_rfi = drift_plane.integrated_rfi();

    return find_components_above_threshold_cpu(doppler_spectrum, dedrifted_rfi, noise_and_thresholds_per_drift, neighborhood);
}
