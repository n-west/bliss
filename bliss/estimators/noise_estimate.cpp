

#include "estimators/noise_estimate.hpp"
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <bland/ops.hpp>
#include <bland/ops_statistical.hpp>

using namespace bliss;

namespace detail {

noise_power noise_power_estimate_stddev(const bland::ndarray &x) {
    noise_power estimated_stats;
    auto        summed_time = bland::mean(x, {0});

    estimated_stats._mean = bland::median(x);

    auto summed_mean     = bland::mean(summed_time);
    estimated_stats._var = std::pow(bland::stddev(summed_time).scalarize<float>(), 2);
    // (bland::sum(bland::square(summed_time - summed_mean)) / (summed_time.numel() - 1)).data_ptr<float>()[0];

    return estimated_stats;
}

noise_power noise_power_estimate_mad(const bland::ndarray &x) {
    noise_power estimated_stats;
    return estimated_stats;
}

} // namespace detail

noise_power bliss::estimate_noise_power(const bland::ndarray &x, noise_power_estimate_options options) {
    noise_power estimated_stats;

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
    // // Using the median as an estimate of the mean is the reason this is a *rough* noise_power function. Spectrum
    // will
    // // often have issues such as a dc offset (shows up as higher power in the DC bin), filter roll-off (shows up as
    // // lowering noise power at the band edges), RFI (high power spurs or spikes). With these kinds of effects
    // present,
    // // the median will often be a better estimate of the noise power mean by minimizing the effect of those outliers.
    // If
    // // those effects are not present the median will be very close to the true mean.
    // estimated_stats.mean = median;

    // // Bessel's correction for sample deviation. Likely doesn't matter given the expected millions
    // // of bins we should have, but is good practice.
    // auto var            = ((x - estimated_stats.mean).square().sum() / (x.size() - 1));
    // estimated_stats.std = std::sqrt(var);

    return estimated_stats;
}

/**
 * This is the masked equivalent of noise power estimate
 */

noise_power bliss::estimate_noise_power(filterbank_data fil_data, noise_power_estimate_options options) {
     noise_power estimated_stats;

    // switch (options.estimator_method) {
    // case noise_power_estimator::MEAN_ABSOLUTE_DEVIATION: {
    //     estimated_stats = detail::noise_power_estimate_mad(x);
    // } break;
    // case noise_power_estimator::STDDEV: {
    //     estimated_stats = detail::noise_power_estimate_stddev(x);
    // } break;
    // default: {
    //     estimated_stats = detail::noise_power_estimate_stddev(x);
    // }
    // }

    return estimated_stats;
}
