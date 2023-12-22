
#include "bland/ndarray.hpp"
#include <drift_search/hit_search.hpp>
#include <drift_search/connected_components.hpp>
#include <drift_search/local_maxima.hpp>

#include <fmt/core.h>
#include <fmt/format.h>
#include <cstdint>

using namespace bliss;

float bliss::compute_signal_threshold(const noise_stats &noise_stats, int64_t integration_length, float snr_threshold) {
    // When the signal amplitude is snr_threshold above the noise floor, we have a 'prehit' (a signal that naively
    // passes a hard threshold) that is when S/N > snr_threshold Given a noise floor estimate of nf, signal amplitude s,
    // noise amplitude n...
    // S = (s - nf)**2
    // N = (n)**2         our estimate has already taken in to account noise floor
    // (s-nf)/(n) > sqrt(snr_threshold)
    // s-nf > n * sqrt(snr_threshold)
    // s > nf + sqrt(N * snr_threshold)
    // Since the noise power was estimate before integration, it also decreases by sqrt of integration length
    float integration_adjusted_noise_power = noise_stats.noise_power() / std::sqrt(integration_length);
    auto  threshold = noise_stats.noise_floor() + std::sqrt(integration_adjusted_noise_power * snr_threshold);
    return threshold;
}

bland::ndarray bliss::hard_threshold_drifts(const bland::ndarray &dedrifted_spectrum,
                                            const noise_stats    &noise_stats,
                                            int64_t               integration_length,
                                            float                 snr_threshold) {

    auto hard_threshold = compute_signal_threshold(noise_stats, integration_length, snr_threshold);

    auto threshold_mask = dedrifted_spectrum > hard_threshold;

    return threshold_mask;
}



std::vector<hit> bliss::hit_search(doppler_spectrum dedrifted_spectrum, noise_stats noise_stats, hit_search_options options) {
    std::vector<hit> hits;

    // auto threshold_mask = hard_threshold_drifts(dedrifted_spectrum.dedrifted_spectrum(),
    //                                             noise_stats,
    //                                             dedrifted_spectrum.integration_length(),
    //                                             snr_threshold);

    // // Now run connected components....
    // auto components = find_components_in_binary_mask(threshold_mask);
    std::vector<component> components;
    if (options.method == hit_search_methods::CONNECTED_COMPONENTS) {
        components = find_components_above_threshold(dedrifted_spectrum, noise_stats, options.snr_threshold, options.neighborhood);
    } else if (options.method == hit_search_methods::LOCAL_MAXIMA) {
        components = find_local_maxima_above_threshold(dedrifted_spectrum, noise_stats, options.snr_threshold, options.neighborhood);
    }

    hits.reserve(components.size());
    // Do we need a "component to hit" for each type of search?
    for (const auto &c : components) {
        // Assume dims size 2 for now :-| (we'll get beam stuff sorted eventually)
        hit this_hit;
        this_hit.rate_index = c.index_max[0];
        this_hit.start_freq_index = c.index_max[1];

        // Start frequency in Hz is bin * Hz/bin
        this_hit.start_freq_MHz = dedrifted_spectrum.fch1() + dedrifted_spectrum.foff() * this_hit.start_freq_index;

        // TODO: sort out better names for these things as we go to whole-cadence integration
        auto drift_freq_span_bins = dedrifted_spectrum.integration_options().low_rate + this_hit.rate_index * dedrifted_spectrum.integration_options().rate_step_size;
        float drift_span_freq_Hz = drift_freq_span_bins * 1e6 * dedrifted_spectrum.foff();

        auto drift_span_time_bins = dedrifted_spectrum.integration_length();
        auto drift_span_time_sec = drift_span_time_bins * dedrifted_spectrum.tsamp();

        this_hit.drift_rate_Hz_per_sec = drift_span_freq_Hz / drift_span_time_sec;

        auto signal_power = std::pow((c.max_integration - noise_stats.noise_floor()), 2);
        auto noise_power = (noise_stats.noise_power()/std::sqrt(dedrifted_spectrum.integration_length()));
        this_hit.snr = signal_power/noise_power;
        // fmt::print("s = {}, n = {} ||| so S / N = {} / {} = {} ", c.max_integration, noise_stats.noise_floor(), signal_power, noise_power, this_hit.snr);

        // At the drift rate with max SNR, find the width of this component
        // We can also integrate signal power over the entire bandwidth / noise power over bandwidth to get
        // a better picture of actual SNR rather than SNR/Hz @ peak
        // This concept of the bandwidth currently doesn't fit well with the local maxima option, but maybe we can come up
        // with something
        this_hit.binwidth = 1;
        int64_t lower_freq_index_at_rate = this_hit.start_freq_index;
        int64_t upper_freq_index_at_rate = this_hit.start_freq_index;
        for (const auto &l : c.locations) {
            if (l[0] == this_hit.rate_index) {
                if (l[1] > upper_freq_index_at_rate) {
                    upper_freq_index_at_rate = l[1];
                } else if (l[1] < lower_freq_index_at_rate) {
                    lower_freq_index_at_rate = l[1];
                }
            }
        }
        this_hit.binwidth = upper_freq_index_at_rate - lower_freq_index_at_rate;
        this_hit.bandwidth = this_hit.binwidth * std::abs(1e6 * dedrifted_spectrum.foff());
        hits.push_back(this_hit);
    }

    return hits;
}
