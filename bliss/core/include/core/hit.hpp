#pragma once

#include "flag_values.hpp"

#include <map>

namespace bliss {
using rfi = std::map<flag_values, uint8_t>; // TODO: not so elegant, but OKish?

struct hit {
    // we need a start time
    int64_t start_freq_index;
    float   start_freq_MHz;
    int64_t rate_index;
    float   drift_rate_Hz_per_sec;
    float   snr;
    int64_t bandwidth;
    double  binwidth;
    rfi     rfi_counts;
};

} // namespace bliss
