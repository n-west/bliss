

#include "estimators/noise_estimate.hpp"
#include <bland/ops.hpp>
#include <bland/ops_statistical.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

using namespace bliss;

namespace detail {

/**
 * Compute noise floor as the mean of the population
 * Compute noise power as the variance of the population
 */
noise_stats noise_power_estimate_stddev(const bland::ndarray &x) {
    noise_stats estimated_stats;

    estimated_stats._noise_floor = bland::mean(x).scalarize<float>();
    estimated_stats._noise_power = bland::stddev(x).scalarize<float>();

    return estimated_stats;
}

/**
 * Compute noise floor as the mean of the population where the mask == 0
 * Compute noise power as the variance of the population where the mask == 0
 */
noise_stats noise_power_estimate_stddev(const bland::ndarray &x, const bland::ndarray &mask) {
    noise_stats estimated_stats;

    estimated_stats._noise_floor = bland::masked_mean(x, mask).scalarize<float>();
    estimated_stats._noise_power = bland::masked_stddev(x, mask).scalarize<float>();

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
    estimated_stats._noise_power   = median_absolute_deviation;

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
bland::ndarray correct_mask(const bland::ndarray &mask) {
    auto unmasked_samples = bland::count_true(mask == 0);
    if (unmasked_samples == 0) {
        // TODO: issue warning & "correct" the mask in some intelligent way
        fmt::format("correct_mask: the mask is completely flagged, so a flagged noise estimate is not possible.");
        /*
         * This is a pretty strange condition where the entire scan is flagged which makes estimating noise using
         * unflagged samples pretty awkward... There's not an obviously right way to handle this and anyone caring
         * about this pipeline output should probably be made aware of it, but it's also not fatal.
         * Known instances of this condition occurring:
         * * Voyager 2020 data from GBT experiences a sudden increase in noise floor/power of ~ 3dB in the B target scan
         *   which generates high spectral kurtosis across the entire band
         *   filename w/in BL: single_coarse_guppi_59046_80354_DIAG_VOYAGER-1_0012.rawspec.0000.h5
         */
        // TODO: we can attempt to correct this, since it's only been known to occur based on SK with high thresholds of
        // ~5 that threshold can be increased. Some ideas:
        // * have an "auto" mode for SK that starts with threshold of 5 and iteratively increases until the whole
        // channel
        //   is not falled
        // * ignore high SK here (or try to identify what flag is causing issues and ignore it)
        // * warn earlier in flagging step
        throw std::runtime_error("correct_mask: the mask is completely flagged");
    }
    auto corrected_mask = bland::copy(mask);
    return mask;
}


/**
 * This is the masked equivalent of noise power estimate
 */
noise_stats bliss::estimate_noise_power(coarse_channel cc_data, noise_power_estimate_options options) {
    noise_stats estimated_stats;
    if (options.masked_estimate) {
        // Check the mask will give us something valid
        auto mask = cc_data.mask();
        cc_data.set_mask(correct_mask(mask));

        switch (options.estimator_method) {
        case noise_power_estimator::MEAN_ABSOLUTE_DEVIATION: {
            estimated_stats = detail::noise_power_estimate_mad(cc_data.data(), mask);
        } break;
        case noise_power_estimator::STDDEV: {
            estimated_stats = detail::noise_power_estimate_stddev(cc_data.data(), mask);
        } break;
        default: {
            estimated_stats = detail::noise_power_estimate_stddev(cc_data.data(), mask);
        }
        }
    } else {
        switch (options.estimator_method) {
        case noise_power_estimator::MEAN_ABSOLUTE_DEVIATION: {
            estimated_stats = detail::noise_power_estimate_mad(cc_data.data());
        } break;
        case noise_power_estimator::STDDEV: {
            estimated_stats = detail::noise_power_estimate_stddev(cc_data.data());
        } break;
        default: {
            estimated_stats = detail::noise_power_estimate_stddev(cc_data.data());
        }
        }
    }
    return estimated_stats;
}

scan bliss::estimate_noise_power(scan fil_data, noise_power_estimate_options options) {
    auto number_coarse_channels = fil_data.get_number_coarse_channels();
    for (auto cc_index = 0; cc_index < number_coarse_channels; ++cc_index) {
        auto cc = fil_data.get_coarse_channel(cc_index);
        auto cc_noise_estimate = estimate_noise_power(*cc, options);
        cc->set_noise_estimate(cc_noise_estimate);
    }
    return fil_data;
}

observation_target bliss::estimate_noise_power(observation_target observations, noise_power_estimate_options options) {
    for (auto &target_scan : observations._scans) {
        target_scan = estimate_noise_power(target_scan, options);
    }
    return observations;
}

cadence bliss::estimate_noise_power(cadence observations, noise_power_estimate_options options) {
    for (auto &obs_target : observations._observations) {
        obs_target = estimate_noise_power(obs_target, options);
    }
    return observations;
}
