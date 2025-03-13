
#include <drift_search/protohit_search.hpp>

#include <drift_search/connected_components.hpp>
#include <drift_search/local_maxima.hpp>

#include <fmt/format.h>

#include <cmath> // sqrt
#include <stdexcept> // runtime_error


using namespace bliss;

std::vector<protohit> bliss::protohit_search(bliss::frequency_drift_plane &drift_plane, int64_t integration_length, noise_stats noise_estimate, hit_search_options options) {

    // The integration_length is only needed to adjust the noise per drift
    // we might be able to get rid of passing that around if we adjust the drift plane
    // to be a correct integration power
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

std::pair<std::vector<protohit>, std::vector<frequency_drift_plane::drift_rate>>
bliss::driftblock_protohit_search(coarse_channel &working_cc, noise_stats noise_estimate, hit_search_options options) {
    std::vector<protohit> protohits;
    std::vector<frequency_drift_plane::drift_rate> drift_rate_info;

    // TODO: the actual work
    
    return std::make_pair(protohits, drift_rate_info);
}

