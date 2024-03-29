#pragma once

#include "hit.hpp"

#include <list>
#include <string>

namespace bliss {

struct event {
    std::list<hit> hits; // hits that contribute to this event
    double         starting_frequency_Hz         = 0;
    float          average_power                 = 0;
    float          average_bandwidth             = 0;
    float          average_snr                   = 0;
    double         average_drift_rate_Hz_per_sec = 0;
    double         event_start_seconds           = 0;
    double         event_end_seconds             = 0;

    std::string repr();
};

} // namespace bliss
