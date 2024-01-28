#pragma once

#include <tuple>

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
    float noise_power() const;

    float noise_amplitude() const;

    float noise_floor() const;

    using state_tuple = std::tuple<float, float>;

    state_tuple get_state() const;

    void set_state(state_tuple);

};

} // namespace bliss