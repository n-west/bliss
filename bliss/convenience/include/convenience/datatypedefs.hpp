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
struct noise_power {
    float _mean;
    float _var;

    public:
    float var() const {
        return _var;
    }

    float stddev() const {
        return std::sqrt(_var);
    }

    float mean() const {
        return _mean;
    }


};

} // namespace bliss