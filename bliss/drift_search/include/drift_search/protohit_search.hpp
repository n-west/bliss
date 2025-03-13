#pragma once
#include "hit_search_options.hpp"

#include <core/protohit.hpp>
#include <core/coarse_channel.hpp>

#include <vector>

namespace bliss {

std::vector<protohit>
protohit_search(bliss::frequency_drift_plane &drift_plane, int64_t integration_length, noise_stats noise_estimate, hit_search_options options = {});

std::pair<std::vector<protohit>, std::vector<frequency_drift_plane::drift_rate>>
driftblock_protohit_search(coarse_channel &working_cc, noise_stats noise_estimate, hit_search_options options = {});

} // namespace bliss
