
#include <drift_search/protohit_search.hpp>

#include <drift_search/connected_components.hpp>
#include <drift_search/local_maxima.hpp>

#include <fmt/format.h>

#include <cmath> // sqrt
#include <stdexcept> // runtime_error


using namespace bliss;

std::vector<protohit> bliss::protohit_search(bliss::frequency_drift_plane &drift_plane, noise_stats noise_estimate, hit_search_options options) {
    auto integration_length = drift_plane.integration_steps();

    std::vector<protohit_drift_info> noise_per_drift;
    noise_per_drift.reserve(drift_plane.drift_rate_info().size());
    for (auto &drift_rate : drift_plane.drift_rate_info()) {
        float integration_adjusted_noise_power = noise_estimate.noise_power() / std::sqrt(integration_length * drift_rate.desmeared_bins);
        noise_per_drift.push_back(protohit_drift_info{.integration_adjusted_noise=integration_adjusted_noise_power});
    }

    auto doppler_spectrum = drift_plane.integrated_drift_plane();
    auto dedrifted_rfi    = drift_plane.integrated_rfi();

    if (doppler_spectrum.dtype() != bland::ndarray::datatype::float32) {
        throw std::runtime_error(
                "protohit_search: dedrifted doppler spectrum was not float which is the only supported datatype.");
    }

    std::vector<protohit> components;
    if (options.method == hit_search_methods::CONNECTED_COMPONENTS) {
        components = find_components_above_threshold(
                doppler_spectrum, dedrifted_rfi, noise_estimate.noise_floor(), noise_per_drift, options.snr_threshold, options.neighbor_l1_dist);
    } else if (options.method == hit_search_methods::LOCAL_MAXIMA) {
        components = find_local_maxima_above_threshold(
                doppler_spectrum, dedrifted_rfi, noise_estimate.noise_floor(), noise_per_drift, options.snr_threshold, options.neighbor_l1_dist);
    }

    return components;
}
