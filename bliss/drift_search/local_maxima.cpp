
#include <drift_search/local_maxima.hpp>

#include "kernels/local_maxima_cpu.hpp"
#include "kernels/local_maxima_cuda.cuh"

using namespace bliss;


std::vector<component> bliss::find_local_maxima_above_threshold(coarse_channel        &dedrifted_coarse_channel,
                                                                float                  snr_threshold,
                                                                std::vector<bland::nd_coords> max_neighborhood) {
    auto noise_stats = dedrifted_coarse_channel.noise_estimate();

    auto drift_plane = dedrifted_coarse_channel.integrated_drift_plane();
    auto integration_length = drift_plane.integration_steps();

    const auto noise_and_thresholds_per_drift = compute_noise_and_snr_thresholds(noise_stats, integration_length, drift_plane.drift_rate_info(), snr_threshold);


    auto doppler_spectrum = drift_plane.integrated_drift_plane();
    auto dedrifted_rfi = drift_plane.integrated_rfi();

    return find_local_maxima_above_threshold_cpu(doppler_spectrum, dedrifted_rfi, noise_and_thresholds_per_drift, max_neighborhood);
}
