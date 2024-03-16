
#include <core/flag_values.hpp>
#include <flaggers/spectral_kurtosis.hpp>
#include <estimators/spectral_kurtosis.hpp>

#include <bland/ops/ops.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

using namespace bliss;

bland::ndarray bliss::flag_spectral_kurtosis(const bland::ndarray &data,
                                             int64_t               N,
                                             int64_t               M,
                                             float                 d,
                                             float                 lower_threshold,
                                             float                 upper_threshold) {
    auto sk = estimate_spectral_kurtosis(data, N, M, d);

    // 2. Threshold & set on channels
    auto rfi = (sk < lower_threshold) * static_cast<uint8_t>(flag_values::low_spectral_kurtosis) +
               (sk > upper_threshold) * static_cast<uint8_t>(flag_values::high_spectral_kurtosis);
    rfi = rfi.unsqueeze(0); // Get the time channel back so we'll broadcast properly
    return rfi;
}

coarse_channel bliss::flag_spectral_kurtosis(coarse_channel cc_data, float lower_threshold, float upper_threshold) {
    auto spectrum_grid = cc_data.data();
    auto rfi_flags     = cc_data.mask();

    // 1. Compute params for SK estimate
    auto M  = spectrum_grid.size(0);
    auto d  = 1;
    auto Fs = std::abs(1.0 / (1e6 * cc_data.foff()));
    auto N  = std::round(cc_data.tsamp() / Fs);

    // 2. Generate SK flag
    auto rfi = flag_spectral_kurtosis(spectrum_grid, N, M, d, lower_threshold, upper_threshold);

    auto accumulated_rfi = rfi_flags + rfi;
    // 3. Store back accumulated rfi
    cc_data.set_mask(accumulated_rfi);

    return cc_data;
}

scan bliss::flag_spectral_kurtosis(scan fil_data, float lower_threshold, float upper_threshold) {
    auto number_coarse_channels = fil_data.get_number_coarse_channels();
    for (auto cc_index = 0; cc_index < number_coarse_channels; ++cc_index) {
        auto cc = fil_data.get_coarse_channel(cc_index);
        *cc = flag_spectral_kurtosis(*cc, lower_threshold, upper_threshold);
    }
    return fil_data;
}

observation_target
bliss::flag_spectral_kurtosis(observation_target observations, float lower_threshold, float upper_threshold) {
    for (auto &filterbank : observations._scans) {
        filterbank = flag_spectral_kurtosis(filterbank, lower_threshold, upper_threshold);
    }
    return observations;
}

cadence bliss::flag_spectral_kurtosis(cadence observations, float lower_threshold, float upper_threshold) {
    for (auto &observation : observations._observations) {
        observation = flag_spectral_kurtosis(observation, lower_threshold, upper_threshold);
    }
    return observations;
}
