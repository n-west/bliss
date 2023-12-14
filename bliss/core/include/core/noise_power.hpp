#pragma once

#include <cmath> // std::sqrt

namespace bliss {


/**
 * noise_power represents estimates of the mean noise power as well as the deviation
 * of noise power. This is typically estimated by calling noise_power_estimate on
 * spectrum data.
 *
 * mean: the expected value of noise power. This is most related to actual noise power by formal definition
 * var: the variance of noise power, i.e., how much we expect an individual sample of noise power to vary by
 */
struct noise_stats {
    float _noise_floor;
    float _noise_power;

    public:
    float noise_power() const {
        return _noise_power;
    }

    float noise_amplitude() const {
        return std::sqrt(_noise_power);
    }

    float noise_floor() const {
        return _noise_floor;
    }


};

} // namespace bliss