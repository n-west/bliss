#pragma once

#include "flag_values.hpp"
#include "protohit.hpp"

#include <map>
#include <string>
#include <tuple>

namespace bliss {

struct hit {
    // we need a start time
    int64_t start_freq_index;
    double  start_freq_MHz;
    double  start_time_sec; // MJD converted to seconds
    double  duration_sec;
    int64_t rate_index;
    double  drift_rate_Hz_per_sec;
    double  power;
    int64_t time_span_steps; // this feels poorly named and maybe should belong next to duration_sec
    float   snr;
    float   bandwidth;
    int64_t binwidth;
    rfi     rfi_counts;
    int64_t coarse_channel_number; // the coarse channel index that this hit was found in

  public:
    std::string repr() const;

    bool operator==(const hit& other) const {
        return start_freq_index == other.start_freq_index &&
               start_freq_MHz == other.start_freq_MHz &&
               start_time_sec == other.start_time_sec &&
               duration_sec == other.duration_sec &&
               rate_index == other.rate_index &&
               drift_rate_Hz_per_sec == other.drift_rate_Hz_per_sec &&
               power == other.power &&
               time_span_steps == other.time_span_steps &&
               snr == other.snr &&
               bandwidth == other.bandwidth &&
               binwidth == other.binwidth &&
               rfi_counts == other.rfi_counts;
    }

    bool operator<(const hit& other) const {
        if (start_freq_index != other.start_freq_index) {
            return start_freq_index < other.start_freq_index;
        }
        if (start_freq_MHz != other.start_freq_MHz) {
            return start_freq_MHz < other.start_freq_MHz;
        }
        if (start_time_sec != other.start_time_sec) {
            return start_time_sec < other.start_time_sec;
        }
        if (duration_sec != other.duration_sec) {
            return duration_sec < other.duration_sec;
        }
        if (rate_index != other.rate_index) {
            return rate_index < other.rate_index;
        }
        if (drift_rate_Hz_per_sec != other.drift_rate_Hz_per_sec) {
            return drift_rate_Hz_per_sec < other.drift_rate_Hz_per_sec;
        }
        if (power != other.power) {
            return power < other.power;
        }
        if (time_span_steps != other.time_span_steps) {
            return time_span_steps < other.time_span_steps;
        }
        if (snr != other.snr) {
            return snr < other.snr;
        }
        if (bandwidth != other.bandwidth) {
            return bandwidth < other.bandwidth;
        }
        if (binwidth != other.binwidth) {
            return binwidth < other.binwidth;
        }
        return rfi_counts < other.rfi_counts;
    }

    using state_tuple = std::tuple<int64_t /*start_freq_index*/,
                                   float /*start_freq_MHz*/,
                                   int64_t /*rate_index*/,
                                   float /*drift_rate_Hz_per_sec*/,
                                   float /*power*/,
                                   float /*time_span_steps*/,
                                   float /*snr*/,
                                   double /*bandwidth*/,
                                   int64_t /*binwidth*/
                                   // rfi /*rfi_counts*/,
                                   >;

    state_tuple get_state() const;

    void set_state(state_tuple state);
};

} // namespace bliss
