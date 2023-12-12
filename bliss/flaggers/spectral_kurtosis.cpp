
#include "flaggers/spectral_kurtosis.hpp"
#include "estimators/spectral_kurtosis.hpp"

#include <bland/ops.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

using namespace bliss;


filterbank_data bliss::flag_spectral_kurtosis(filterbank_data fb_data) {
    auto &spectrum_grid = fb_data.data();

    auto &rfi_flags = fb_data.mask();

    auto M  = spectrum_grid.size(0);
    auto d  = 1;
    auto Fs = 1.0 / (1e6 * fb_data.foff());

    auto N = fb_data.tsamp() / Fs;

    // 1. Compute spectral kurtosis along each channel
    // SK = (M N d + 1)/(M-1) * (M S_2 / S_1^2 -1)
    //
    // * d is 1
    // * N is the number of spectrograms already averaged per spectra we receive
    // * M is the number of spectra in this population to estimate kurtosis over (commonly 8, 16, or 32)

    auto sk = spectral_kurtosis(spectrum_grid, N, M, 1.0);

    // 2. Threshold & set on channels
    // auto mask_above = sk > 50;
    // auto mask_below = sk < .05;

    return fb_data;
}
