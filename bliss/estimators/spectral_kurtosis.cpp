#include "estimators/spectral_kurtosis.hpp"

#include <bland/ops/ops.hpp>

#include <fmt/format.h>

using namespace bliss;

bland::ndarray bliss::estimate_spectral_kurtosis(const bland::ndarray &spectrum_grid, int32_t N, int32_t M, float d) {
    fmt::print("INFO: spec kurtosis with M={}, N={}, d={}\n", M, N, d);
    auto s1 = bland::square(bland::sum(spectrum_grid, {0}));
    auto s2 = bland::sum(bland::square(spectrum_grid), {0});
    auto sk = ((M * N * d + 1) / (M - 1)) * (M * (s2 / s1) - 1);
    return sk;
}

bland::ndarray bliss::estimate_spectral_kurtosis(coarse_channel &cc_data, float d) {
    const bland::ndarray spectrum_grid = cc_data.data();

    // 1. Compute spectral kurtosis along each channel
    auto M  = spectrum_grid.size(0);
    auto Fs = std::abs(1.0 / (1e6 * cc_data.foff()));
    auto N  = std::round(cc_data.tsamp() / Fs);

    auto sk = estimate_spectral_kurtosis(spectrum_grid, N, M, d);

    return sk;
}

