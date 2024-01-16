
#include <drift_search/event_search.hpp>

#include <core/cadence.hpp>

#include <fmt/format.h>
#include <cstdint>

using namespace bliss;

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
