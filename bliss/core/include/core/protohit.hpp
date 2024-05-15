#pragma once

#include "flag_values.hpp"

#include <map>
#include <vector>

namespace bliss {

using rfi = std::map<flag_values, uint8_t>; // TODO: not so elegant, but OKish?

struct freq_drift_coord {
    int64_t drift_index=0;
    int64_t frequency_channel=0;
};

struct protohit {
    freq_drift_coord index_max;
    freq_drift_coord index_center;
    float snr;
    float max_integration;
    float desmeared_noise;
    int binwidth;
    std::vector<freq_drift_coord> locations;
    rfi rfi_counts;
};


// This is a cuda-compatible version (lacking std::vector) of a protohit
struct device_protohit {
    freq_drift_coord index_max;
    freq_drift_coord index_center;
    float snr;
    float max_integration;
    float desmeared_noise;
    // freq_drift_coord* locations;
    int binwidth;
    // rfi rfi_counts;
    uint8_t low_sk_count;
    uint8_t high_sk_count;
    // -1 indicates an invalid protohit, 0 indicates valid, > 0 indicates another protohit that is better
    int invalidated_by=-1;
};

} // namespace bliss