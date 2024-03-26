#pragma once
#include <drift_search/hit_search_options.hpp>

#include <core/flag_values.hpp>
#include <core/hit.hpp>
#include <core/noise_power.hpp>
#include <core/scan.hpp>


#include <bland/stride_helper.hpp>

#include <list>

namespace bliss {

struct protohit {
    std::vector<bland::nd_coords> locations;
    float                  max_integration = std::numeric_limits<float>::lowest();
    float                  desmeared_noise;
    rfi                    rfi_counts;
    bland::nd_coords       index_max;
};

std::vector<protohit>
protohit_search(coarse_channel &dedrifted_coarse_channel, hit_search_options options = {});


} // namespace bliss
