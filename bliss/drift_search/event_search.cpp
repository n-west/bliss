
#include <drift_search/event_search.hpp>

#include <core/cadence.hpp>

#include <fmt/format.h>
#include <cstdint>

using namespace bliss;

constexpr float seconds_per_day = 24 * 60 * 60;

constexpr float freq_localization_weight = 0.01f;
constexpr float drift_error_weight       = 10.0f;
constexpr float snr_difference_weight    = 0.0f;
constexpr float eps                      = 1e-8;

float distance_func(const hit &a, const hit &b) {
    auto snr_difference   = std::abs(a.snr - b.snr);
    auto power_difference = std::abs(a.power - b.power);
    auto drift_difference = std::abs(a.drift_rate_Hz_per_sec - b.drift_rate_Hz_per_sec) /
                            (eps + a.drift_rate_Hz_per_sec * a.drift_rate_Hz_per_sec +
                             b.drift_rate_Hz_per_sec * b.drift_rate_Hz_per_sec);
    auto drift_error = drift_difference * drift_difference;

    auto first_sample_time = std::min(a.start_time_sec, b.start_time_sec);
    auto last_sample_time  = std::max(a.start_time_sec + b.duration_sec, b.start_time_sec + b.duration_sec);
    auto rendezvous_time   = (last_sample_time + first_sample_time) / 2;

    auto a_time_to_rendezvous = rendezvous_time - a.start_time_sec;
    auto b_time_to_rendezvous = rendezvous_time - b.start_time_sec;

    auto a_rendezvous_frequency = a.start_freq_MHz * 1e6 + a.drift_rate_Hz_per_sec * a_time_to_rendezvous;
    auto b_rendezvous_frequency = b.start_freq_MHz * 1e6 + b.drift_rate_Hz_per_sec * b_time_to_rendezvous;

    auto rendezvous_frequency_difference = std::abs(a_rendezvous_frequency - b_rendezvous_frequency);

    auto distance = freq_localization_weight * rendezvous_frequency_difference + drift_error_weight * drift_error +
                    snr_difference_weight * snr_difference;
    return distance;
}

/**
 * A container to expose the distance_function through a built-in commutative cache
 *
 * Have not benchmarked the distance function, but presumably it will only get more complex over time
 */
struct hit_distance {
    struct hit_pair_comparator {
        bool operator()(const std::pair<hit, hit> &p1, const std::pair<hit, hit> &p2) const {
            if (p1.first == p2.first && p1.second == p2.second) {
                return false;
            }
            if (p1.first == p2.second && p1.second == p2.first) {
                return false;
            }
            return p1 < p2;
        }
    };

    std::map<std::pair<hit, hit>, float, hit_pair_comparator> distance_cache;

    float operator()(hit p1, hit p2) {
        auto pair = std::make_pair(p1, p2);
        if (distance_cache.find(pair) != distance_cache.end()) {
            return distance_cache.at(pair);
        } else {
            auto d               = distance_func(p1, p2);
            distance_cache[pair] = d;
            return d;
        }
    }
};


std::vector<event> bliss::event_search(cadence cadence_with_hits) {
    hit_distance       distance;
    std::vector<event> detected_events;

    // TODO: should we find "matches" between same-scan as part of filtering

    // assume the on_target is the 0-indexed target inside the cadence. This can be adjusted
    // or we could even loop over multiple on_targets in the future, so keep it as a variable
    const int64_t on_target_index = 0;
    auto          on_target_obs   = cadence_with_hits._observations[on_target_index];
    std::vector<std::list<hit>> on_scan_hits;
    for (auto &scan : on_target_obs._scans) {
        on_scan_hits.push_back(scan.hits());
    }

    // Flatten all OFF scans (order doesn't matter)
    std::vector<scan> off_scans;
    for (size_t ii = 0; ii < cadence_with_hits._observations.size(); ++ii) {
        if (ii != on_target_index) {
            for (auto &target_scan : cadence_with_hits._observations[ii]._scans) {
                off_scans.push_back(target_scan);
            }
        }
    }


    for (size_t on_scan_index = 0; on_scan_index < on_target_obs._scans.size(); ++on_scan_index) {
        auto starting_scan           = on_target_obs._scans[on_scan_index];
        // For every hit, look through hits in subsequent scans for hit with the lowest distance
        for (const auto &starting_hit : on_scan_hits[on_scan_index]) {
            event candidate_event;
            candidate_event.hits.push_back(starting_hit);
            candidate_event.average_power                 = starting_hit.power;
            candidate_event.average_snr                   = starting_hit.snr;
            candidate_event.average_drift_rate_Hz_per_sec = starting_hit.drift_rate_Hz_per_sec;
            candidate_event.event_start_seconds           = starting_scan.tstart() * seconds_per_day;
            candidate_event.starting_frequency_Hz         = starting_hit.start_freq_MHz * 1e6;
            candidate_event.event_end_seconds =
                    starting_scan.tstart() * seconds_per_day + starting_scan.tduration_secs();

            for (size_t matching_scan_index = on_scan_index + 1; matching_scan_index < on_target_obs._scans.size();
                 ++matching_scan_index) {
                auto &candidate_scan = on_target_obs._scans[matching_scan_index];

                // Assume there is a good matching hit in this scan, extend the candidate event start / end
                auto hypothetical_start =
                        std::min(candidate_event.event_start_seconds, seconds_per_day * candidate_scan.tstart());
                auto hypothetical_end = std::max(candidate_event.event_end_seconds,
                                                 seconds_per_day * candidate_scan.tstart() + candidate_scan.tsamp());
                auto rendezvous_time =
                        (hypothetical_start + hypothetical_end) / 2; // a point in time to extrapolate both hits to

                auto candidate_event_time_to_rendezvous = rendezvous_time - candidate_event.event_start_seconds;
                auto scan_time_to_rendezvous            = rendezvous_time - candidate_scan.tstart() * seconds_per_day;
                auto event_rendezvous_frequency =
                        candidate_event.starting_frequency_Hz +
                        candidate_event.average_drift_rate_Hz_per_sec * candidate_event_time_to_rendezvous;

                float best_distance_to_a_hit = std::numeric_limits<float>::max();
                auto& hits_to_check          = on_scan_hits[matching_scan_index];

                std::list<hit>::iterator best_matching_hit;
                for (auto candidate_matching_hit = hits_to_check.begin(); candidate_matching_hit != hits_to_check.end();
                     ++candidate_matching_hit) {
                    auto lowest_distance_to_event = std::numeric_limits<float>::max();
                    for (const auto hit_in_event : candidate_event.hits) {
                        auto d = distance(hit_in_event, *candidate_matching_hit);
                        if (d < lowest_distance_to_event) {
                            lowest_distance_to_event = d;
                        }
                    }
                    if (lowest_distance_to_event < best_distance_to_a_hit) {
                        best_distance_to_a_hit = lowest_distance_to_event;
                        best_matching_hit      = candidate_matching_hit;
                    }
                }
                if (best_distance_to_a_hit < 50 /* Made up number that looks good when staring at distances */) {
                    candidate_event.hits.push_back(*best_matching_hit);
                    hits_to_check.erase(best_matching_hit);
                    // TODO: fix this
                    // candidate_scan.hits(hits_to_check); // update the hits in the scan with the removed matches
                }
            }

            int times_event_in_off = 0;
            for (auto off_scan : off_scans) {
                for (auto off_hit : off_scan.hits()) {
                    float distance_to_event_hits = 0;
                    for (auto event_hit : candidate_event.hits) {
                        distance_to_event_hits += distance(off_hit, event_hit);
                    }
                    if (distance_to_event_hits / candidate_event.hits.size() < 50) {
                        times_event_in_off += 1;
                        fmt::print("INFO: Event was found in an off scan\n");
                    }
                }
            }
            if (candidate_event.hits.size() > 1 && times_event_in_off == 0) {
                // TODO: check if the candidate event seems like a good event (so far, making sure > 1 hit matches)
                // I can do some of this when adding hits to the candidate event
                candidate_event.average_drift_rate_Hz_per_sec = 0;
                candidate_event.average_power = 0;
                candidate_event.average_snr = 0;
                candidate_event.average_bandwidth = 0;
                for (auto &hit_in_event : candidate_event.hits) {
                    candidate_event.average_drift_rate_Hz_per_sec += hit_in_event.drift_rate_Hz_per_sec;
                    candidate_event.average_power += hit_in_event.power;
                    candidate_event.average_snr += hit_in_event.snr;
                    candidate_event.average_bandwidth += hit_in_event.bandwidth;
                }
                candidate_event.average_drift_rate_Hz_per_sec =
                        candidate_event.average_drift_rate_Hz_per_sec / candidate_event.hits.size();
                candidate_event.average_power = candidate_event.average_power / candidate_event.hits.size();
                candidate_event.average_bandwidth = candidate_event.average_bandwidth / candidate_event.hits.size();
                candidate_event.average_snr   = candidate_event.average_snr / candidate_event.hits.size();
                fmt::print("INFO: Average SNR of this candidate event is {} and drift is {}\n",
                           candidate_event.average_snr,
                           candidate_event.average_drift_rate_Hz_per_sec);

                // candidate_event.starting_frequency = candidate_event.average_drift; // I don't have a good way to
                // compute this right now
                detected_events.emplace_back(candidate_event);
            }
        }
        // If we have a valid event, erase all of the matching hits
    }

    return detected_events;
}
