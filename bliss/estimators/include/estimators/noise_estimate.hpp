
#pragma once

#include <core/noise_power.hpp>
#include <core/filterbank_data.hpp>
#include <core/cadence.hpp>
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
     * Estimate noise power statistics of the array using the given method. Noise stats available are
     * 
     * * noise floor: the estimated magnitude of noise samples.
     * * noise power: the noise power (how much noise deviates from the noise floor)
     * 
    */
    [[nodiscard]] noise_stats estimate_noise_power(const bland::ndarray &x, noise_power_estimate_options options);

    /**
     * Estimate noise power statistics of filterbank_data (this can include a flagged/masked estimate)
    */
    [[nodiscard]] noise_stats estimate_noise_power(filterbank_data fil_data, noise_power_estimate_options options);

    /**
     * Estimate noise power statistics of an observation target which may include multiple discrete "views" of the target
     * 
     * The returned value is an observation_target with valid noise_stats estimate per observation
    */
    [[nodiscard]] observation_target estimate_noise_power(observation_target observations, noise_power_estimate_options options);

    /**
     * Estimate noise power statistics for each observation target in a cadence
     * 
     * The returned value is a cadence with each observation of each target filled in with a valid noise_stat estimate
     * 
     * This API may change to return a type better suited for cadences
    */
    [[nodiscard]] cadence estimate_noise_power(cadence observations, noise_power_estimate_options options);

} // namespace bliss

