#pragma once

#include <core/cadence.hpp>
#include <core/hit.hpp>
#include <core/event.hpp>

#include <list>

namespace bliss {

/**
 * Find *events* which are hits correlated across time
 * // TODO: there's a better way to structure this so we can make more sense of time / on/off by reusing the
 * cadence/observation_target structure/hierarchy
 */
std::vector<event> event_search(cadence cadence_with_hits);

} // namespace bliss
