
#include "bland/ndarray.hpp"
#include <drift_search/connected_components.hpp>
#include <drift_search/hit_search.hpp>
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

std::list<hit> bliss::hit_search(scan dedrifted_scan, hit_search_options options) {
    std::list<hit> hits;

    std::vector<component> components;
    if (options.method == hit_search_methods::CONNECTED_COMPONENTS) {
        components = find_components_above_threshold(dedrifted_scan, options.snr_threshold, options.neighborhood);
    } else if (options.method == hit_search_methods::LOCAL_MAXIMA) {
        components = find_local_maxima_above_threshold(dedrifted_scan, options.snr_threshold, options.neighborhood);
    }
    auto noise_stats = dedrifted_scan.noise_estimate();

    // Do we need a "component to hit" for each type of search?
    for (const auto &c : components) {
        // Assume dims size 2 for now :-| (we'll get beam stuff sorted eventually)
        hit this_hit;
        this_hit.rate_index       = c.index_max[0];
        this_hit.rfi_counts       = c.rfi_counts;
        this_hit.start_freq_index = c.index_max[1];

        // Start frequency in Hz is bin * Hz/bin
        auto freq_offset        = dedrifted_scan.foff() * this_hit.start_freq_index;
        this_hit.start_freq_MHz = dedrifted_scan.fch1() + freq_offset;

        auto drift_freq_span_bins = dedrifted_scan.dedoppler_options().low_rate +
                                    this_hit.rate_index * dedrifted_scan.dedoppler_options().rate_step_size;
        float drift_span_freq_Hz = drift_freq_span_bins * 1e6 * dedrifted_scan.foff();

        auto drift_span_time_bins = dedrifted_scan.integration_length();
        auto drift_span_time_sec  = drift_span_time_bins * dedrifted_scan.tsamp();

        this_hit.drift_rate_Hz_per_sec = drift_span_freq_Hz / drift_span_time_sec;

        auto signal_power = std::pow((c.max_integration - noise_stats.noise_floor()), 2);
        auto noise_power  = (noise_stats.noise_power() / std::sqrt(dedrifted_scan.integration_length()));
        this_hit.snr      = signal_power / noise_power;

        // At the drift rate with max SNR, find the width of this component
        // We can also integrate signal power over the entire bandwidth / noise power over bandwidth to get
        // a better picture of actual SNR rather than SNR/Hz @ peak
        // This concept of the bandwidth currently doesn't fit well with the local maxima option, but maybe we can come
        // up with something
        this_hit.binwidth                = 1;
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
        this_hit.binwidth  = upper_freq_index_at_rate - lower_freq_index_at_rate;
        this_hit.bandwidth = this_hit.binwidth * std::abs(1e6 * dedrifted_scan.foff());

        // TODO: consider if we should focus hits on the "center" or keep it on strongest frequency that triggered
        // the hit. For local maxima they should be similar but can deviate quite a bit for connected components
        auto center_bin         = (upper_freq_index_at_rate + lower_freq_index_at_rate) / 2;
        freq_offset             = dedrifted_scan.foff() * center_bin;
        this_hit.start_freq_MHz = dedrifted_scan.fch1() + freq_offset;
        this_hit.start_time_sec = dedrifted_scan.tstart() * 24 * 60 * 60; // convert MJD to seconds since MJ
        this_hit.duration_sec   = dedrifted_scan.tsamp() * dedrifted_scan.integration_length();
        hits.push_back(this_hit);
    }

    return hits;
}

observation_target bliss::hit_search(observation_target dedrifted_target, hit_search_options options) {
    for (auto &dedrifted_scan : dedrifted_target._scans) {
        auto hits = hit_search(dedrifted_scan, options);
        dedrifted_scan.hits(hits);
    }
    return dedrifted_target;
}

cadence bliss::hit_search(cadence dedrifted_cadence, hit_search_options options) {
    for (auto &obs_target : dedrifted_cadence._observations) {
        obs_target = hit_search(obs_target, options);
    }
    return dedrifted_cadence;
}
