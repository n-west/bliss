#pragma once

#include <core/cadence.hpp>
#include <core/hit.hpp>

#include <list>

namespace bliss {

struct event {
    std::list<hit> hits; // hits that contribute to this event
    float          starting_frequency_Hz         = 0;
    float          average_power                 = 0;
    float          average_snr                   = 0;
    float          average_drift_rate_Hz_per_sec = 0;
    double         event_start_seconds           = 0;
    double         event_end_seconds             = 0;
};

/**
 * Find *events* which are hits correlated across time
 * // TODO: there's a better way to structure this so we can make more sense of time / on/off by reusing the
 * cadence/observation_target structure/hierarchy
 */
std::vector<event> event_search(cadence cadence_with_hits);

} // namespace bliss
