
#include "flaggers/kurtosis.hpp"
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <bland/ops.hpp>
#include <bland/ops_statistical.hpp>
#include <iostream>

using namespace bliss;


bland::ndarray flag_spectral_kurtosis(const bland::ndarray &spectrum_grid) {
    auto rfi_flags = bland::ndarray(spectrum_grid.shape());

    // 1. Compute kurtosis along each channel
    auto per_channel_kurtosis = bland::standardized_moment(spectrum_grid, 4, {0});
    

    // 2. Threshold & set on channels

    return rfi_flags;
}
