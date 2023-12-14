

#include "estimators/noise_estimate.hpp"
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <bland/ops.hpp>
#include <bland/ops_statistical.hpp>

using namespace bliss;

namespace detail {

noise_power noise_power_estimate_stddev(const bland::ndarray &x) {
    noise_power estimated_stats;

    estimated_stats._mean = bland::mean(x).scalarize<float>();
    estimated_stats._var  = bland::var(x).scalarize<float>();

    return estimated_stats;
}

noise_power noise_power_estimate_stddev(const bland::ndarray &x, const bland::ndarray &mask) {
    noise_power estimated_stats;

    estimated_stats._mean = bland::masked_mean(x, mask).scalarize<float>();
    estimated_stats._var  = std::pow(bland::masked_stddev(x, mask).scalarize<float>(), 2);
    // estimated_stats._var = 0;

    return estimated_stats;
}

noise_power noise_power_estimate_mad(const bland::ndarray &x) {
    noise_power estimated_stats;

    // bland needs some TLC to do this Right (TM)
    estimated_stats._mean = bland::median(x);
    // median absolute deviation is median(|Xi - median|)
    auto stddev          = bland::median(bland::abs(x - estimated_stats._mean));
    estimated_stats._var = std::pow(stddev, 2);

    return estimated_stats;
}

noise_power noise_power_estimate_mad(const bland::ndarray &x, const bland::ndarray &mask) {

    throw std::runtime_error("masked noise power estimation with mad is not implemented yet");
    noise_power estimated_stats;

    // bland needs some TLC to do this Right (TM) and until then
    // it's pretty hard to do the masked_median
    estimated_stats._mean = bland::median(x);
    // median absolute deviation is median(|Xi - median|)
    auto stddev          = bland::median(bland::abs(x - estimated_stats._mean));
    estimated_stats._var = std::pow(stddev, 2);

    return estimated_stats;
}

} // namespace detail

noise_power bliss::estimate_noise_power(const bland::ndarray &x, noise_power_estimate_options options) {
    noise_power estimated_stats;

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

noise_power bliss::estimate_noise_power(filterbank_data fil_data, noise_power_estimate_options options) {
    noise_power estimated_stats;
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
