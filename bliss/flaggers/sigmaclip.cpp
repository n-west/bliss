#include <flaggers/sigmaclip.hpp>

#include <core/flag_values.hpp>

#include <bland/ops/ops.hpp>

#include <fmt/core.h>

using namespace bliss;

bland::ndarray bliss::flag_sigmaclip(const bland::ndarray &data, int max_iter, float low, float high) {

    constexpr float eps = 1e-6;
    auto            rfi = bland::zeros(data.shape(), data.dtype(), data.device());

    auto [mean, stddev] = bland::masked_mean_stddev(data, rfi);
    for (int iter = 0; iter < max_iter; ++iter) {
        // fmt::print("iter {}:  mean={}    std={}\n", iter, mean.scalarize<float>(), stddev.scalarize<float>());
        rfi = (data < (mean - stddev * low)) + (data > (mean + stddev * high));

        auto [new_mean, new_std] = bland::masked_mean_stddev(data, rfi);
        if (std::abs((new_mean - mean).scalarize<float>()) < eps &&
            std::abs((new_std - stddev).scalarize<float>()) < eps) {
            break;
        } else {
            mean   = new_mean;
            stddev = new_std;
        }
    }

    return rfi * static_cast<uint8_t>(flag_values::sigma_clip);
}

coarse_channel bliss::flag_sigmaclip(coarse_channel cc_data, int max_iter, float low, float high) {

    bland::ndarray spectrum_grid = cc_data.data();
    bland::ndarray rfi_flags     = cc_data.mask();

    // 2. Generate SK flag
    auto rfi = flag_sigmaclip(spectrum_grid, max_iter, low, high);

    auto accumulated_rfi = rfi_flags + rfi; // a | operator would be more appropriate

    // 3. Store back accumulated rfi
    cc_data.set_mask(accumulated_rfi);

    return cc_data;
}

scan bliss::flag_sigmaclip(scan fil_data, int max_iter, float low, float high) {
    fil_data.add_coarse_channel_transform([max_iter, low, high](coarse_channel cc) {
        return flag_sigmaclip(cc, max_iter, low, high);
    });
    return fil_data;
}

observation_target bliss::flag_sigmaclip(observation_target observations, int max_iter, float low, float high) {
    for (auto &filterbank : observations._scans) {
        filterbank = flag_sigmaclip(filterbank, max_iter, low, high);
    }
    return observations;
}

cadence bliss::flag_sigmaclip(cadence observations, int max_iter, float low, float high) {
    for (auto &observation : observations._observations) {
        observation = flag_sigmaclip(observation, max_iter, low, high);
    }
    return observations;
}
