
#include <drift_search/protohit_search.hpp>

#include <drift_search/connected_components.hpp>
#include <drift_search/local_maxima.hpp>

using namespace bliss;

std::vector<protohit> bliss::protohit_search(coarse_channel &dedrifted_coarse_channel, hit_search_options options) {

    // If neighborhood is empty, fill it with L1 distance=1
    auto neighborhood = options.neighborhood;
    if (neighborhood.empty()) {
        neighborhood = {
                {-1, 0},
                {1, 0},
                {0, -1},
                {0, 1},
        };
    }

    dedrifted_coarse_channel.set_device("cpu");
    dedrifted_coarse_channel.push_device();

    auto noise_stats = dedrifted_coarse_channel.noise_estimate();

    auto drift_plane        = dedrifted_coarse_channel.integrated_drift_plane();
    auto integration_length = drift_plane.integration_steps();

    const auto noise_and_thresholds_per_drift = compute_noise_and_snr_thresholds(
            noise_stats, integration_length, drift_plane.drift_rate_info(), options.snr_threshold);

    auto doppler_spectrum = drift_plane.integrated_drift_plane();
    auto dedrifted_rfi    = drift_plane.integrated_rfi();

    std::vector<protohit> components;
    if (options.method == hit_search_methods::CONNECTED_COMPONENTS) {
        components = find_components_above_threshold(
                doppler_spectrum, dedrifted_rfi, noise_and_thresholds_per_drift, neighborhood);
    } else if (options.method == hit_search_methods::LOCAL_MAXIMA) {
        components = find_local_maxima_above_threshold(
                doppler_spectrum, dedrifted_rfi, noise_and_thresholds_per_drift, neighborhood);
    }

    return components;
}
