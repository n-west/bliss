
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
