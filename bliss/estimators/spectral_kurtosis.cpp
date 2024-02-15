#include "estimators/spectral_kurtosis.hpp"

#include <bland/ops.hpp>

#include <fmt/format.h>

using namespace bliss;

bland::ndarray bliss::estimate_spectral_kurtosis(const bland::ndarray &spectrum_grid, int64_t N, int64_t M, float d) {
    fmt::print("INFO: spec kurtosis with M={} and N={}\n", M, N);
    auto s1 = bland::square(bland::sum(spectrum_grid, {0}));
    auto s2 = bland::sum(bland::square(spectrum_grid), {0});
    auto sk = ((M * N * d + 1) / (M - 1)) * (M * (s2 / s1) - 1);
    return sk;
}

bland::ndarray bliss::estimate_spectral_kurtosis(filterbank_data &fil_data) {
    const auto spectrum_grid = fil_data.data();

    // 1. Compute spectral kurtosis along each channel
    auto M  = spectrum_grid.size(0);
    auto d  = 1;
    auto Fs = std::abs(1.0 / (1e6 * fil_data.foff()));
    auto N  = std::round(fil_data.tsamp() / Fs);

    auto sk = estimate_spectral_kurtosis(spectrum_grid, N, M, 1.0);

    return sk;
}
