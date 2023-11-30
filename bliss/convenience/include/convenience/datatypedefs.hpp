#pragma once

#include <cmath> // std::sqrt

namespace bliss {

/**
 * Convenience typdef for a flexibly-sized 2d array of floats.
*/
// using row_major_f32array = Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
// using row_major_ui8array = Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

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