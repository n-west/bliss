#pragma once

#include <core/hit.hpp>
#include <core/cadence.hpp>

#include <vector>

namespace bliss {

struct event {
    // event() = default;
    // event(std::vector<hit> hits) : hits(hits) {}
    std::vector<hit> hits; // hits that contribute to this event
    float starting_frequency = 0;
    float average_power = 0;
    float average_snr = 0;
    float average_drift_rate_Hz_per_sec = 0;
};

/**
 * Find *events* which are hits correlated across time
 * // TODO: there's a better way to structure this so we can make more sense of time / on/off by reusing the
 * cadence/observation_target structure/hierarchy
 */
std::vector<event> event_search(cadence cadence_with_hits);

} // namespace bliss
