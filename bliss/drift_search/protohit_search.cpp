
#include <drift_search/protohit_search.hpp>

#include <drift_search/compute_drift_rates.hpp>
#include <drift_search/integrate_drifts.hpp>
#include <drift_search/connected_components.hpp>
#include <drift_search/local_maxima.hpp>

#include <fmt/format.h>

#include <cmath> // sqrt
#include <stdexcept> // runtime_error


using namespace bliss;

std::vector<protohit> bliss::protohit_search(bliss::frequency_drift_plane &drift_plane, int64_t integration_length, noise_stats noise_estimate, hit_search_options options) {

    // The integration_length is only needed to adjust the noise per drift
    // we might be able to get rid of passing that around if we adjust the integrated power
    // at the time of integration rather than adjusting the noise power
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
    std::vector<protohit> all_protohits;
    std::vector<frequency_drift_plane::drift_rate> drift_rate_info;

    // Outline of the work to do:
    // 1) For each drift block we need to process:
    //   a) Integrate the drift block
    //   b) Search for protohits in the integrated drift block, passing context from the previous drift block
    //
    // We can probably save some of the final protohit collection for the very end
    // Let's just pretend we're doing *one* of the hit search options right now (let's pick local maxima) so we need
    // integrate over each drift

    // TODO: does it make sense to extract this to its own function?

    auto compute_device = working_cc.device();
    auto integration_options = options.integration_options;
    
    auto drift_rates = compute_drifts(working_cc.ntsteps(), working_cc.foff(), working_cc.tsamp(), integration_options);

    auto number_drift_blocks = drift_rates.size() / working_cc.ntsteps(); // For now break the work in to multiples of timesteps so it's like a natural tt

    auto drifts_per_block = drift_rates.size() / number_drift_blocks;
    // TODO: check that after any int arithmetic drifts_per_block * number_drift_blocks will capture all drifts

    auto data = working_cc.data();
    auto mask = working_cc.mask();
    for (int drift_block = 0; drift_block < number_drift_blocks; ++drift_block) {
        auto block_drifts = std::vector<frequency_drift_plane::drift_rate>(drift_rates.begin() + drift_block * drifts_per_block,
                                                                          drift_rates.begin() + (drift_block + 1) * drifts_per_block);

        fmt::print("Working on drift block {}/{} with drift ranges {} : {} ({} : {})\n",
                   drift_block,
                   number_drift_blocks,
                   drift_block * drifts_per_block,
                   (drift_block + 1) * drifts_per_block,
                   block_drifts.front().drift_rate_Hz_per_sec,
                   block_drifts.back().drift_rate_Hz_per_sec);
        auto drift_plane = integrate_drifts(data, mask, block_drifts, integration_options);
        auto protohits = protohit_search(drift_plane, working_cc.ntsteps(), noise_estimate, options);

        fmt::print("Found {} protohits in drift block {}/{}\n", protohits.size(), drift_block, number_drift_blocks);
        all_protohits.insert(all_protohits.end(), protohits.begin(), protohits.end());
        
    }
    

    return std::make_pair(all_protohits, drift_rates);
}

