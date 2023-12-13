
#include "flaggers/spectral_kurtosis.hpp"
#include "estimators/spectral_kurtosis.hpp"

#include <bland/ops.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

using namespace bliss;


filterbank_data bliss::flag_spectral_kurtosis(filterbank_data fb_data, float lower_threshold, float upper_threshold) {
    auto spectrum_grid = fb_data.data();
    auto &rfi_flags = fb_data.mask();

    // 1. Compute spectral kurtosis along each channel
    auto M  = spectrum_grid.size(0);
    auto d  = 1;
    auto Fs = std::abs(1.0 / (1e6 * fb_data.foff()));
    auto N = std::round(fb_data.tsamp() / Fs);

    auto sk = spectral_kurtosis(spectrum_grid, N, M, 1.0);

    // 2. Threshold & set on channels
    auto rfi = (sk < lower_threshold) + (sk > upper_threshold);
    rfi = rfi.unsqueeze(0);
    
    // TODO: use a slice to store in-place
    bland::copy(rfi, rfi_flags);

    return fb_data;
}
