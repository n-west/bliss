
#pragma once

#include "convenience/datatypedefs.hpp"
#include <bland/ndarray.hpp>

namespace bliss
{

    /**
     * Which method to use to estimate noise power mean and variance of a drift spectrum detection plane.
     * 
     * TURBO_SETI estimate the same way turbo_seti estimates noise statistics. This uses
     * a sum down the time dimension. The "mean" is the median of the resulting sum. The
     * variance is the variance of the population with extremes in the low and high 5% removed
     * 
     * RV_TRANSFORM estimate the the noise power mean using the median of all spectra and
     * variance using 1/(N-1) sum(x - median). This swaps the mean (first moment) for the
     * median since that is used as an estimate for true noise mean. Bessel's correction is used
     * for variance estimate.
     * 
     * MEAN_ABSOLUTE_DEVIATION estimate the noise power mean using MAD (Median Absolute Deviation)
     * which is the median of absolute magnitude of deviation from the median
    */
    enum class noise_power_estimator {
        TURBO_SETI,
        RV_TRANSFORM,
        MEAN_ABSOLUTE_DEVIATION,
    };

    /**
     * Estimate noise power statistics using the given method
     * 
    */
    [[nodiscard]] noise_power noise_power_estimate(const bland::ndarray &x, noise_power_estimator estimator_method);

    [[nodiscard]] noise_power noise_power_estimate(const bland::ndarray &x, const bland::ndarray &mask, noise_power_estimator estimator_method);

} // namespace bliss

