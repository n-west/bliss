
#include <core/hit.hpp>

using namespace bliss;

hit::state_tuple bliss::hit::get_state() const {
    return std::make_tuple(start_freq_index,
                           start_freq_MHz,
                           rate_index,
                           drift_rate_Hz_per_sec,
                           power,
                           time_span_steps,
                           snr,
                           bandwidth,
                           binwidth
                           // rfi_counts,
    );
}

void bliss::hit::set_state(state_tuple state) {
    start_freq_index      = std::get<0>(state);
    start_freq_MHz        = std::get<1>(state);
    rate_index            = std::get<2>(state);
    drift_rate_Hz_per_sec = std::get<3>(state);
    power                 = std::get<4>(state);
    time_span_steps       = std::get<5>(state);
    snr                   = std::get<6>(state);
    bandwidth             = std::get<7>(state);
    binwidth              = std::get<8>(state);
}
