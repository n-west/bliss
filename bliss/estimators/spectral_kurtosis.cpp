#include "estimators/spectral_kurtosis.hpp"

#include <bland/ops.hpp>

using namespace bliss;

bland::ndarray bliss::spectral_kurtosis(const bland::ndarray &spectrum_grid, float N, float M, float d) {
    auto s1 = bland::square(bland::sum(spectrum_grid, {0}));
    auto s2 = bland::sum(bland::square(spectrum_grid), {0});
    auto sk = (M * N * d + 1) / (M - 1) * (M * s2 / s1 - 1);
    return sk;
}
