
#include "core/noise_power.hpp"
#include <cmath> // std::sqrt

using namespace bliss;

float bliss::noise_stats::noise_power() const {
    return _noise_power;
}

float bliss::noise_stats::noise_amplitude() const {
    return std::sqrt(_noise_power);
}

float bliss::noise_stats::noise_floor() const {
    return _noise_floor;
}

noise_stats::state_tuple bliss::noise_stats::get_state() const {
    return std::make_tuple(_noise_floor, _noise_power);
}

void bliss::noise_stats::set_state(noise_stats::state_tuple state) {
    _noise_floor = std::get<0>(state);
    _noise_power = std::get<1>(state);
}
