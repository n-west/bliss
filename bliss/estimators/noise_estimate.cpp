

#include "estimators/noise_estimate.hpp"
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <bland/ops.hpp>
#include <bland/ops_statistical.hpp>

using namespace bliss;

namespace detail {

/**
 * Compute noise floor as the mean of the population
 * Compute noise power as the variance of the population
 */
noise_stats noise_power_estimate_stddev(const bland::ndarray &x) {
    noise_stats estimated_stats;

    estimated_stats._noise_floor = bland::mean(x).scalarize<float>();
    estimated_stats._noise_power = bland::var(x).scalarize<float>();

    return estimated_stats;
}

/**
 * Compute noise floor as the mean of the population where the mask == 0
 * Compute noise power as the variance of the population where the mask == 0
 */
noise_stats noise_power_estimate_stddev(const bland::ndarray &x, const bland::ndarray &mask) {
    noise_stats estimated_stats;

    estimated_stats._noise_floor = bland::masked_mean(x, mask).scalarize<float>();
    estimated_stats._noise_power = std::pow(bland::masked_stddev(x, mask).scalarize<float>(), 2);

    return estimated_stats;
}

/**
 * Compute noise floor using the median over the population
 * Compute noise power using the Median Absolute Deviation (MAD) over the population
 */
noise_stats noise_power_estimate_mad(const bland::ndarray &x) {
    noise_stats estimated_stats;

    // bland needs some TLC to do this Right (TM)
    estimated_stats._noise_floor = bland::median(x);
    // median absolute deviation is median(|Xi - median|)
    auto median_absolute_deviation = bland::median(bland::abs(x - estimated_stats._noise_floor));
    estimated_stats._noise_power   = std::pow(median_absolute_deviation, 2);

    return estimated_stats;
}

/**
 * TODO: this is equivalent to the non masked noise_power_estimate_mad until improved support
 * for the mask operation can be done
 */
noise_stats noise_power_estimate_mad(const bland::ndarray &x, const bland::ndarray &mask) {

    throw std::runtime_error("masked noise power estimation with mad is not implemented yet");
    noise_stats estimated_stats;

    // bland needs some TLC to do this Right (TM) and until then
    // it's pretty hard to do the masked_median
    estimated_stats._noise_floor = bland::median(x);
    // median absolute deviation is median(|Xi - median|)
    auto stddev                  = bland::median(bland::abs(x - estimated_stats._noise_floor));
    estimated_stats._noise_power = std::pow(stddev, 2);

    return estimated_stats;
}

} // namespace detail

noise_stats bliss::estimate_noise_power(const bland::ndarray &x, noise_power_estimate_options options) {
    noise_stats estimated_stats;

    if (options.masked_estimate) {
        fmt::print("WARN: Requested a masked noise estimate, but calling without a mask. This may be an error in the "
                   "future.\n");
    }

    switch (options.estimator_method) {
    case noise_power_estimator::MEAN_ABSOLUTE_DEVIATION: {
        estimated_stats = detail::noise_power_estimate_mad(x);
    } break;
    case noise_power_estimator::STDDEV: {
        estimated_stats = detail::noise_power_estimate_stddev(x);
    } break;
    default: {
        estimated_stats = detail::noise_power_estimate_stddev(x);
    }
    }
    return estimated_stats;
}

/**
 * validate & correct a flag mask
 * 
 * If the mask has no free flags
*/
bland::ndarray correct_mask(bland::ndarray mask) {
    auto unmasked_samples = bland::count_true(mask == 0);
    if (unmasked_samples == 0) {
        // TODO: issue warning & "correct" the mask in some intelligent way
        throw std::runtime_error("correct_mask: the mask is completely flagged");
    }
    // auto corrected_mask = bland::copy(mask);
    return mask;
}

/**
 * This is the masked equivalent of noise power estimate
 */

noise_stats bliss::estimate_noise_power(filterbank_data fil_data, noise_power_estimate_options options) {
    noise_stats estimated_stats;
    if (options.masked_estimate) {
        // Check the mask will give us something valid
        auto &mask = fil_data.mask();
        mask = correct_mask(mask);
        switch (options.estimator_method) {
        case noise_power_estimator::MEAN_ABSOLUTE_DEVIATION: {
            estimated_stats = detail::noise_power_estimate_mad(fil_data.data(), mask);
        } break;
        case noise_power_estimator::STDDEV: {
            estimated_stats = detail::noise_power_estimate_stddev(fil_data.data(), mask);
        } break;
        default: {
            estimated_stats = detail::noise_power_estimate_stddev(fil_data.data(), mask);
        }
        }
    } else {
        switch (options.estimator_method) {
        case noise_power_estimator::MEAN_ABSOLUTE_DEVIATION: {
            estimated_stats = detail::noise_power_estimate_mad(fil_data.data());
        } break;
        case noise_power_estimator::STDDEV: {
            estimated_stats = detail::noise_power_estimate_stddev(fil_data.data());
        } break;
        default: {
            estimated_stats = detail::noise_power_estimate_stddev(fil_data.data());
        }
        }
    }
    return estimated_stats;
}

observation_target bliss::estimate_noise_power(observation_target observations, noise_power_estimate_options options) {
    for (auto &target_scan : observations._scans) {
        auto scan_noise_estimate = estimate_noise_power(target_scan, options);
        target_scan.noise_estimate(scan_noise_estimate);
    }
    return observations;
}

cadence bliss::estimate_noise_power(cadence observations, noise_power_estimate_options options) {
    for (auto &obs_target : observations._observations) {
        // cadence_noise_stats.push_back(estimate_noise_power(observation, options));
        obs_target = estimate_noise_power(obs_target, options);
    }
    return observations;
}
