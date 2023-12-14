
#pragma once

#include <core/noise_power.hpp>
#include <core/filterbank_data.hpp>
#include <bland/ndarray.hpp>

namespace bliss
{

    /**
     * Which method to use to estimate noise power mean and variance of a drift spectrum detection plane.
     * 
     * 
     * MEAN_ABSOLUTE_DEVIATION estimate the noise power mean using MAD (Median Absolute Deviation)
     * which is the median of absolute magnitude of deviation from the median
    */
    enum class noise_power_estimator {
        STDDEV,
        MEAN_ABSOLUTE_DEVIATION,
    };

    struct noise_power_estimate_options {
        noise_power_estimator estimator_method;
        bool masked_estimate = true;
    };

    /**
     * Estimate noise power statistics using the given method
     * 
    */
    [[nodiscard]] noise_power estimate_noise_power(const bland::ndarray &x, noise_power_estimate_options options);

    [[nodiscard]] noise_power estimate_noise_power(filterbank_data fil_data, noise_power_estimate_options options);

} // namespace bliss

