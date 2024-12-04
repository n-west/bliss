#pragma once

#include <bland/ndarray_deferred.hpp>

#include <tuple>
#include <string>

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

    public:
    float noise_power();
    void set_noise_power(bland::ndarray noise_power);

    float noise_amplitude();

    float noise_floor();
    void set_noise_floor(bland::ndarray noise_floor);

    std::string repr() const;

    // using state_tuple = std::tuple<float, float>;

    // state_tuple get_state() const;

    // void set_state(state_tuple);

    protected:
    bland::ndarray _noise_floor;
    bland::ndarray _noise_power;
};

} // namespace bliss