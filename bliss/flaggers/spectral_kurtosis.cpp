
#include "flaggers/spectral_kurtosis.hpp"
#include "estimators/spectral_kurtosis.hpp"

#include <bland/ops.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

using namespace bliss;


filterbank_data bliss::flag_spectral_kurtosis(filterbank_data fb_data, float lower_threshold, float upper_threshold) {
    auto &spectrum_grid = fb_data.data();

    auto &rfi_flags = fb_data.mask();

    // 1. Compute spectral kurtosis along each channel
    // SK = (M N d + 1)/(M-1) * (M S_2 / S_1^2 -1)
    //
    // * d is 1
    // * N is the number of spectrograms already averaged per spectra we receive
    // * M is the number of spectra in this population to estimate kurtosis over (commonly 8, 16, or 32)
    auto M  = spectrum_grid.size(0);
    auto d  = 1;
    auto Fs = 1.0 / (1e6 * fb_data.foff());
    auto N = fb_data.tsamp() / Fs;

    auto sk = spectral_kurtosis(spectrum_grid, N, M, 1.0);

    // 2. Threshold & set on channels
    auto rfi = (sk < lower_threshold) + (sk > upper_threshold);

    bland::copy(rfi_flags.unsqueeze(0), rfi);
    // auto mask_above = sk > 50.0f;
    // auto mask_below = sk < .05f;

    return fb_data;
}
