#pragma once

#include <core/hit.hpp>
#include <core/cadence.hpp>

#include <vector>

namespace bliss {

struct event {
    // event() = default;
    // event(std::vector<hit> hits) : hits(hits) {}
    std::vector<hit> hits; // hits that contribute to this event
};

/**
 * Find *events* which are hits correlated across time
 * // TODO: there's a better way to structure this so we can make more sense of time / on/off by reusing the
 * cadence/observation_target structure/hierarchy
 */
std::vector<event> event_search(cadence cadence_with_hits);

} // namespace bliss
