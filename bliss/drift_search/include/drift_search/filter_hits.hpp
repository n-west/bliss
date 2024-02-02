#pragma once

#include <core/cadence.hpp>
#include <core/hit.hpp>
#include <core/scan.hpp>

#include <list>

namespace bliss {

struct filter_options {};

std::list<hit> filter_hits(std::list<hit>, filter_options options);

scan filter_hits(scan scan_with_hits, filter_options options);

observation_target filter_hits(observation_target scans_with_hits, filter_options options);

cadence filter_hits(cadence cadence_with_hits, filter_options options);

} // namespace bliss
