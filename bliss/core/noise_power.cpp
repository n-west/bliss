
#include "core/noise_power.hpp"

#include <bland/bland.hpp>

#include <fmt/format.h>

#include <cmath> // std::sqrt

using namespace bliss;

float bliss::noise_stats::noise_power() {
    return bland::to(_noise_power, "cpu").scalarize<float>();
}

void bliss::noise_stats::set_noise_power(bland::ndarray power) {
    _noise_power = power;
}

float bliss::noise_stats::noise_amplitude() {
    return std::sqrt(bland::to(_noise_power, "cpu").scalarize<float>());
}

float bliss::noise_stats::noise_floor() {
    return bland::to(_noise_floor, "cpu").scalarize<float>();
}

void bliss::noise_stats::set_noise_floor(bland::ndarray nf) {
    _noise_floor = nf;
}

std::string bliss::noise_stats::repr() const {
    auto r = fmt::format("noise_floor={} + noise_power={}", _noise_floor.repr(), _noise_power.repr());
    return r;
}

// noise_stats::state_tuple bliss::noise_stats::get_state() const {
//     return std::make_tuple(_noise_floor, _noise_power);
// }

// void bliss::noise_stats::set_state(noise_stats::state_tuple state) {
//     _noise_floor = std::get<0>(state);
//     _noise_power = std::get<1>(state);
// }
