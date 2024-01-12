
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

scan bliss::hit_search(scan dedrifted_scan, hit_search_options options) {
    std::vector<hit> hits;

    std::vector<component> components;
    if (options.method == hit_search_methods::CONNECTED_COMPONENTS) {
        components = find_components_above_threshold(dedrifted_scan, options.snr_threshold, options.neighborhood);
    } else if (options.method == hit_search_methods::LOCAL_MAXIMA) {
        components = find_local_maxima_above_threshold(dedrifted_scan, options.snr_threshold, options.neighborhood);
    }
    auto noise_stats = dedrifted_scan.noise_estimate();

    hits.reserve(components.size());
    // Do we need a "component to hit" for each type of search?
    for (const auto &c : components) {
        // Assume dims size 2 for now :-| (we'll get beam stuff sorted eventually)
        hit this_hit;
        this_hit.rate_index       = c.index_max[0];
        this_hit.rfi_counts       = c.rfi_counts;
        this_hit.start_freq_index = c.index_max[1];

        // Start frequency in Hz is bin * Hz/bin
        this_hit.start_freq_MHz = dedrifted_scan.fch1() + dedrifted_scan.foff() * this_hit.start_freq_index;

        // TODO: sort out better names for these things as we go to whole-cadence integration
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
        hits.push_back(this_hit);
    }

    dedrifted_scan.hits(hits);

    return dedrifted_scan;
}

observation_target
bliss::hit_search(observation_target dedrifted_target, hit_search_options options) {
    for (auto &dedrifted_scan : dedrifted_target._scans) {
        dedrifted_scan = hit_search(dedrifted_scan, options);
    }
    return dedrifted_target;
}


cadence bliss::hit_search(cadence dedrifted_cadence, hit_search_options options) {
    for (auto &obs_target : dedrifted_cadence._observations) {
        obs_target = hit_search(obs_target, options);
    }
    return dedrifted_cadence;
}

std::vector<event> bliss::event_search(cadence cadence_with_hits) {
    std::vector<event> detected_events;

    // assume the on_target is the 0-indexed target inside the cadence. This can be adjusted
    // or we could even loop over multiple on_targets in the future, so keep it as a variable
    const int64_t        on_target_index = 0;
    auto                 on_target_obs   = cadence_with_hits._observations[on_target_index];
    std::vector<int64_t> cadence_ndindex{static_cast<int64_t>(cadence_with_hits._observations.size()), 0};

    // // Check every combination of hits from each scan for matches
    // std::vector<int64_t> on_target_hits_ndindex{static_cast<int64_t>(on_target_obs._scans.size()), 0};
    // // TODO: this is for a hit that is present in *all* scans, what if we want to allow missing one?
    // int64_t hit_perumtations = 1;
    // for (size_t on_scan_index = 0; on_scan_index < on_target_obs._scans.size(); ++on_scan_index) {
    //     hit_perumtations = hit_perumtations * std::max<int64_t>(1, on_target_obs._scans[on_scan_index].hits().size());
    // }
    // for (size_t hit_perm_index = 0; hit_perm_index < hit_perumtations; ++hit_perm_index) {

        

    //     // Increment that index...
    //     for (size_t scan_index = 0; scan_index < on_target_obs._scans.size(); ++scan_index) {
    //         on_target_hits_ndindex[scan_index] += 1;
    //         if (on_target_hits_ndindex[scan_index] < on_target_obs._scans[scan_index].hits().size()) {
    //             break;
    //         } else {
    //             on_target_hits_ndindex[scan_index] = 0;
    //         }
    //     }
    // }

    for (size_t on_scan_index = 0; on_scan_index < on_target_obs._scans.size(); ++on_scan_index) {
        auto starting_scan = on_target_obs._scans[on_scan_index];
        auto candidate_starting_hits = starting_scan.hits();
        // For every hit, look through hits in subsequent scans trying to match them
        for (size_t starting_hit_index = 0; starting_hit_index < candidate_starting_hits.size(); ++starting_hit_index) {
            auto starting_hit = candidate_starting_hits[starting_hit_index];
            event candidate_event;
            candidate_event.hits.push_back(starting_hit);
            for (size_t matching_scan_index = on_scan_index; matching_scan_index < on_target_obs._scans.size(); ++matching_scan_index) {
                auto candidate_scan = on_target_obs._scans[matching_scan_index];
                auto hits_to_check = candidate_scan.hits();
                auto time_between_scans = starting_scan.tstart() - candidate_scan.tstart();
                for (size_t candidate_hit_index = 0; candidate_hit_index < hits_to_check.size(); ++candidate_hit_index) {
                    auto candidate_matching_hit = hits_to_check[candidate_hit_index];

                    auto extrapolated_start_frequency = candidate_matching_hit.start_freq_MHz + candidate_matching_hit.drift_rate_Hz_per_sec * time_between_scans;
                    float max_drift_rate_error = starting_hit.drift_rate_Hz_per_sec * 0.1f; // 10% drift rate error, is there a better error bound?
                    // allow being off by 2 channel widths + the min/max drift error
                    auto channel_frequency_match_error = std::abs(starting_scan.foff()) + std::abs(candidate_scan.foff()) + std::abs(max_drift_rate_error * time_between_scans);

                    if (std::abs(starting_hit.start_freq_MHz - extrapolated_start_frequency) < channel_frequency_match_error &&
                        std::abs(starting_hit.drift_rate_Hz_per_sec - candidate_matching_hit.drift_rate_Hz_per_sec) < max_drift_rate_error) {
                        // We're going to assume these are the same signal, which means we have a candidate hit
                        fmt::print("Building an event between [{}, {}] and [{}][{}]\n", on_scan_index, starting_hit_index, matching_scan_index, candidate_hit_index);
                        candidate_event.hits.push_back(candidate_matching_hit);
                    }
                }
            }
        }
    }

    return detected_events;
}
