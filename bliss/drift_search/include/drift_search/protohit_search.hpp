#pragma once
#include "hit_search_options.hpp"

#include <core/protohit.hpp>
#include <core/coarse_channel.hpp>

#include <vector>

namespace bliss {

std::vector<protohit>
protohit_search(coarse_channel &dedrifted_coarse_channel, hit_search_options options = {});

} // namespace bliss
