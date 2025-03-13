#pragma once
#include "hit_search_options.hpp"

#include <core/protohit.hpp>
#include <core/coarse_channel.hpp>

#include <vector>

namespace bliss {

std::vector<protohit>
protohit_search(bliss::frequency_drift_plane &drift_plane, noise_stats noise_estimate, hit_search_options options = {});

std::vector<protohit>
driftblock_protohit_search(bliss::frequency_drift_plane &drift_plane, noise_stats noise_estimate, hit_search_options options = {});

} // namespace bliss
