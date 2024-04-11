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

    std::vector<bland::nd_coords> neighborhood = {
            // clang-format off
   {-7,  7}, {-6,  7}, {-5,  7}, {-4,  7}, {-3,  7},{-2,  7},  {-1, 7},  {0, 7},  {1, 7}, {2,  7}, {3,  7}, {4,  7}, {5,  7}, {6,  7}, {7,  7},
   {-7,  6}, {-6,  6}, {-5,  6}, {-4,  6}, {-3,  6},{-2,  6},  {-1, 6},  {0, 6},  {1, 6}, {2,  6}, {3,  6}, {4,  6}, {5,  6}, {6,  6}, {7,  6},
   {-7,  5}, {-6,  5}, {-5,  5}, {-4,  5}, {-3,  5},{-2,  5},  {-1, 5},  {0, 5},  {1, 5}, {2,  5}, {3,  5}, {4,  5}, {5,  5}, {6,  5}, {7,  5},
   {-7,  4}, {-6,  4}, {-5,  4}, {-4,  4}, {-3,  4},{-2,  4},  {-1, 4},  {0, 4},  {1, 4}, {2,  4}, {3,  4}, {4,  4}, {5,  4}, {6,  4}, {7,  4},
   {-7,  3}, {-6,  3}, {-5,  3}, {-4,  3}, {-3,  3},{-2,  3},  {-1, 3},  {0, 3},  {1, 3}, {2,  3}, {3,  3}, {4,  3}, {5,  3}, {6,  3}, {7,  3},
   {-7,  2}, {-6,  2}, {-5,  2}, {-4,  2}, {-3,  2},{-2,  2},  {-1, 2},  {0, 2},  {1, 2}, {2,  2}, {3,  2}, {4,  2}, {5,  2}, {6,  2}, {7,  2},
   {-7,  1}, {-6,  1}, {-5,  1}, {-4,  1}, {-3,  1},{-2,  1},  {-1, 1},  {0, 1},  {1, 1}, {2,  1}, {3,  1}, {4,  1}, {5,  1}, {6,  1}, {7,  1},
   {-7,  0}, {-6,  0}, {-5,  0}, {-4,  0}, {-3,  0},{-2,  0},  {-1, 0},  /* X  */ {1, 0}, {2,  0}, {3,  0}, {4,  0}, {5,  0}, {6,  0}, {7,  0},
   {-7, -1}, {-6, -1}, {-5, -1}, {-4, -1}, {-3, -1},{-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1}, {3, -1}, {4, -1}, {5, -1}, {6, -1}, {7, -1},
   {-7, -2}, {-6, -2}, {-5, -2}, {-4, -2}, {-3, -2},{-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}, {3, -2}, {4, -2}, {5, -2}, {6, -2}, {7, -2},
   {-7, -3}, {-6, -3}, {-5, -3}, {-4, -3}, {-3, -3},{-2, -3}, {-1, -3}, {0, -3}, {1, -3}, {2, -3}, {3, -3}, {4, -3}, {5, -3}, {6, -3}, {7, -3},
   {-7, -4}, {-6, -4}, {-5, -4}, {-4, -4}, {-3, -4},{-2, -4}, {-1, -4}, {0, -4}, {1, -4}, {2, -4}, {3, -4}, {4, -4}, {5, -4}, {6, -4}, {7, -4},
   {-7, -5}, {-6, -5}, {-5, -5}, {-4, -5}, {-3, -5},{-2, -5}, {-1, -5}, {0, -5}, {1, -5}, {2, -5}, {3, -5}, {4, -5}, {5, -5}, {6, -5}, {7, -5},
   {-7, -6}, {-6, -6}, {-5, -6}, {-4, -6}, {-3, -6},{-2, -6}, {-1, -6}, {0, -6}, {1, -6}, {2, -6}, {3, -6}, {4, -6}, {5, -6}, {6, -6}, {7, -6},
   {-7, -7}, {-6, -7}, {-5, -7}, {-4, -7}, {-3, -7},{-2, -7}, {-1, -7}, {0, -7}, {1, -7}, {2, -7}, {3, -7}, {4, -7}, {5, -7}, {6, -7}, {7, -7},
            // clang-format on
    };

    bool detach_graph = true;
};

}