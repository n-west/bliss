

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
 * Compute noise floor over all observations of a target where the mask == 0 using population mean as the noise floor
 * Compute noise power over all observations of a target where the mask == 0 using population variance as the noise
 * power
 */
observation_target noise_power_estimate_stddev(observation_target &observations, bool use_mask = true) {
    auto updated_observations = observations;

    for (size_t nn = 0; nn < observations._filterbanks.size(); ++nn) {
        auto        x = observations._filterbanks[nn].data();
        noise_stats estimated_stats;
        if (use_mask) {
            auto mask         = observations._filterbanks[nn].mask();
            auto mean         = bland::masked_mean(x, mask).scalarize<float>();
            auto squared_mean = bland::masked_mean(bland::square(x), mask).scalarize<float>();

            estimated_stats._noise_floor = mean;
            estimated_stats._noise_power = squared_mean - mean * mean;
        } else {
            auto mean         = bland::mean(x).scalarize<float>();
            auto squared_mean = bland::mean(bland::square(x)).scalarize<float>();

            estimated_stats._noise_floor = mean;
            estimated_stats._noise_power = squared_mean - mean * mean;
        }
        updated_observations._filterbanks[nn].noise_estimates(estimated_stats);
    }

    return updated_observations;
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
    auto stddev                  = bland::median(bland::abs(x - estimated_stats._noise_floor));
    estimated_stats._noise_power = std::pow(stddev, 2);

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

/**
 * TODO: think about a concat type of operation to make this work
 */
observation_target noise_power_estimate_mad(observation_target &observations, bool use_mask = true) {
    auto updated_observations = observations;

    for (size_t nn = 0; nn < observations._filterbanks.size(); ++nn) {
        auto        x = observations._filterbanks[nn].data();
        noise_stats estimated_stats;
        if (use_mask) {
            throw std::runtime_error("masked noise power estimation with mad is not implemented yet");
        } else {
            estimated_stats._noise_floor = bland::median(x);
            auto dev                     = bland::median(bland::abs(x - estimated_stats._noise_floor));
            estimated_stats._noise_power = std::pow(dev, 2);
        }
        updated_observations._filterbanks[nn].noise_estimates(estimated_stats);
    }

    return updated_observations;
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
 * This is the masked equivalent of noise power estimate
 */

noise_stats bliss::estimate_noise_power(filterbank_data fil_data, noise_power_estimate_options options) {
    noise_stats estimated_stats;
    if (options.masked_estimate) {
        switch (options.estimator_method) {
        case noise_power_estimator::MEAN_ABSOLUTE_DEVIATION: {
            estimated_stats = detail::noise_power_estimate_mad(fil_data.data(), fil_data.mask());
        } break;
        case noise_power_estimator::STDDEV: {
            estimated_stats = detail::noise_power_estimate_stddev(fil_data.data(), fil_data.mask());
        } break;
        default: {
            estimated_stats = detail::noise_power_estimate_stddev(fil_data.data(), fil_data.mask());
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
    auto updated_observations = observation_target();
    switch (options.estimator_method) {
    case noise_power_estimator::MEAN_ABSOLUTE_DEVIATION: {
        updated_observations = detail::noise_power_estimate_mad(observations, options.masked_estimate);
    } break;
    case noise_power_estimator::STDDEV: {
        updated_observations = detail::noise_power_estimate_stddev(observations, options.masked_estimate);
    } break;
    default: {
        updated_observations = detail::noise_power_estimate_stddev(observations, options.masked_estimate);
    }
    }

    return updated_observations;
}

cadence bliss::estimate_noise_power(cadence observations, noise_power_estimate_options options) {
    auto updated_cadence = cadence();
    for (auto &observation : observations._observations) {
        // cadence_noise_stats.push_back(estimate_noise_power(observation, options));
        updated_cadence._observations.push_back(estimate_noise_power(observation, options));
    }
    return updated_cadence;
}
