#pragma once

#include "core/integrate_drifts_options.hpp"

#include <vector>

namespace bliss {

enum class hit_search_methods { CONNECTED_COMPONENTS, LOCAL_MAXIMA };

struct hit_search_options {
    hit_search_methods method = hit_search_methods::CONNECTED_COMPONENTS;

    /**
     * threshold (linear SNR) that integrated power must be above to be considered a hit
     */
    float snr_threshold = 10.0f;

    int neighbor_l1_dist = 7;

    /**
     * Whether to search in increments of a drift block or generate the whole dedrift plane at once
     * 
     * Generating the entire dedrift plane is nicer for plotting and debugging but consumes more memory
     * which may limit the effective drift rate range that can be searched.
     */
    bool iterative = true;
    integrate_drifts_options integration_options;

};

}