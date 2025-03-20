#pragma once

#include <core/cadence.hpp>
#include <core/hit.hpp>
#include <core/scan.hpp>

#include <list>

namespace bliss {

struct filter_options {
    bool filter_zero_drift = true;

    bool filter_sigmaclip = true;
    float minimum_percent_sigmaclip = 0.1;

    bool filter_high_sk = false;
    float minimum_percent_high_sk = 0.1;

    bool filter_low_sk = false;
    float maximum_percent_low_sk = 0.1;
};

std::list<hit> filter_hits(std::list<hit>, filter_options options);

coarse_channel filter_hits(coarse_channel cc_with_hits, filter_options options);

scan filter_hits(scan scan_with_hits, filter_options options);

observation_target filter_hits(observation_target scans_with_hits, filter_options options);

cadence filter_hits(cadence cadence_with_hits, filter_options options);

} // namespace bliss
