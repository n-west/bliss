#pragma once

#include "flag_values.hpp"

#include <bland/ndarray.hpp>

#include <cstdint>

namespace bliss {

struct integrate_drifts_options {
    // integrate_drifts_options(bool desmear, int64_t low_rate, int64_t high_rate, int64_t rate_step_size);

    // spectrum_sum_method method  = spectrum_sum_method::LINEAR_ROUND;
    bool desmear = true;
    // TODO properly optionize drift ranges
    int64_t low_rate       = -64;
    int64_t high_rate      = 64;
    int64_t rate_step_size = 1;
};


/**
 * a container to track how much rfi of each kind was involved in each drift integration
 */
struct integrated_flags {
    bland::ndarray filter_rolloff;
    bland::ndarray low_spectral_kurtosis;
    bland::ndarray high_spectral_kurtosis;
    bland::ndarray magnitude;
    bland::ndarray sigma_clip;
    integrated_flags(int64_t drifts, int64_t channels, bland::ndarray::dev device = default_device) :
            filter_rolloff(bland::ndarray({drifts, channels}, 0, bland::ndarray::datatype::uint8, device)),
            low_spectral_kurtosis(bland::ndarray({drifts, channels}, 0, bland::ndarray::datatype::uint8, device)),
            high_spectral_kurtosis(bland::ndarray({drifts, channels}, 0, bland::ndarray::datatype::uint8, device)),
            magnitude(bland::ndarray({drifts, channels}, 0, bland::ndarray::datatype::uint8, device)),
            sigma_clip(bland::ndarray({drifts, channels}, 0, bland::ndarray::datatype::uint8, device)) {}
};

} // namespace bliss
