#pragma once

#include <bland/stride_helper.hpp>

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

    bool detach_graph = true;
};

}