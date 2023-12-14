#include "estimators/spectral_kurtosis.hpp"

#include <bland/ops.hpp>

#include <fmt/format.h>

using namespace bliss;

bland::ndarray bliss::estimate_spectral_kurtosis(const bland::ndarray &spectrum_grid, int64_t N, int64_t M, float d) {
    fmt::print("spec kurtosis with M={} and N={}\n", M, N);
    auto s1 = bland::square(bland::sum(spectrum_grid, {0}));
    auto s2 = bland::sum(bland::square(spectrum_grid), {0});
    auto sk = (M * N * d + 1) / (M - 1) * (M * s2 / s1 - 1);
    return sk;
}
