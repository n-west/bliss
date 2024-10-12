#include <flaggers/spectral_kurtosis.hpp>

#include <estimators/spectral_kurtosis.hpp>
#include <core/flag_values.hpp>

#include <bland/ops/ops.hpp>

// #include <fmt/core.h>
// #include <fmt/ranges.h>

#include <cmath> // std::round

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

coarse_channel bliss::flag_spectral_kurtosis(coarse_channel cc_data, float lower_threshold, float upper_threshold, float d) {

    auto cc_ptr = std::make_shared<coarse_channel>(cc_data);

    auto deferred_accumulated_rfi = bland::ndarray_deferred([cc_data = cc_ptr, lower_threshold, upper_threshold, d]() {
        bland::ndarray spectrum_grid = cc_data->data();
        bland::ndarray rfi_flags     = cc_data->mask();

        // 1. Compute params for SK estimate
        auto M  = spectrum_grid.size(0);
        auto Fs = std::abs(1.0 / (1e6 * cc_data->foff()));
        auto N  = std::round(cc_data->tsamp() / Fs);

        // 2. Generate SK flag
        auto rfi = flag_spectral_kurtosis(spectrum_grid, N, M, d, lower_threshold, upper_threshold);

        auto accumulated_rfi = rfi_flags + rfi; // a | operator would be more appropriate
        return accumulated_rfi;
    });

    // 3. Store back accumulated rfi
    cc_data.set_mask(deferred_accumulated_rfi);

    return cc_data;
}

scan bliss::flag_spectral_kurtosis(scan fil_data, float lower_threshold, float upper_threshold, float d) {
    auto number_coarse_channels = fil_data.get_number_coarse_channels();
    for (auto cc_index = 0; cc_index < number_coarse_channels; ++cc_index) {
        auto cc = fil_data.read_coarse_channel(cc_index);
        *cc     = flag_spectral_kurtosis(*cc, lower_threshold, upper_threshold, d);
    }
    return fil_data;
}

observation_target
bliss::flag_spectral_kurtosis(observation_target observations, float lower_threshold, float upper_threshold, float d) {
    for (auto &filterbank : observations._scans) {
        filterbank = flag_spectral_kurtosis(filterbank, lower_threshold, upper_threshold, d);
    }
    return observations;
}

cadence bliss::flag_spectral_kurtosis(cadence observations, float lower_threshold, float upper_threshold, float d) {
    for (auto &observation : observations._observations) {
        observation = flag_spectral_kurtosis(observation, lower_threshold, upper_threshold, d);
    }
    return observations;
}
