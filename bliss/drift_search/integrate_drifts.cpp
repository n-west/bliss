
#include "drift_search/integrate_drifts.hpp"
#include "kernels/drift_integration_bland.hpp"
#include "kernels/drift_integration_cpu.hpp"
#if BLISS_CUDA
#include "kernels/drift_integration_cuda.cuh"
#endif

#include <bland/bland.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

using namespace bliss;

std::vector<frequency_drift_plane::drift_rate> compute_drifts(int time_steps, double foff, double tsamp, integrate_drifts_options options) {
    auto maximum_drift_span = time_steps - 1;

    // Convert the drift options to specific drift info
    auto number_drifts = (options.high_rate - options.low_rate) / options.rate_step_size;
    std::vector<frequency_drift_plane::drift_rate> drift_rate_info;
    drift_rate_info.reserve(number_drifts);
    for (int drift_index = 0; drift_index < number_drifts; ++drift_index) {
        // Allow the options to represent either bin counts or Hz/sec using physical units
        // Drift in number of channels over the entire time extent
        auto drift_channels = options.low_rate + drift_index * options.rate_step_size;
        frequency_drift_plane::drift_rate rate;
        rate.index_in_plane = drift_index;
        rate.drift_channels_span = drift_channels;

        // The actual slope of that drift (number channels / time)
        auto m = static_cast<float>(drift_channels) / static_cast<float>(maximum_drift_span);

        rate.drift_rate_slope = m;
        rate.drift_rate_Hz_per_sec = m * foff * 1e6 / tsamp;
        // If a single time step crosses more than 1 channel, there is smearing over multiple channels
        auto smeared_channels = std::round(std::abs(m));

        int desmear_binwidth = 1;
        if (options.desmear) {
            desmear_binwidth = std::max(1.0F, smeared_channels);
        }
        rate.desmeared_bins = desmear_binwidth;

        drift_rate_info.push_back(rate);
    }
    return drift_rate_info;
}

// bland::ndarray bliss::integrate_drifts(const bland::ndarray &spectrum_grid, integrate_drifts_options options) {
//     auto compute_device = spectrum_grid.device();
//     auto drifts = compute_drifts(spectrum_grid.size(0), 1, 1, options);
// }

coarse_channel bliss::integrate_drifts(coarse_channel cc_data, integrate_drifts_options options) {
    auto compute_device = cc_data.device();

    auto drifts = compute_drifts(cc_data.ntsteps(), cc_data.foff(), cc_data.tsamp(), options);

    fmt::print("INFO: Searching drift rates from {}Hz/sec to {}Hz/sec\n",
               drifts.front().drift_rate_Hz_per_sec,
               drifts.back().drift_rate_Hz_per_sec);

    auto cc_copy = std::make_shared<coarse_channel>(cc_data);
    if (compute_device.device_type == kDLCPU) {
        auto integrated_dedrift = [cc_data = cc_copy, drifts, options]() {
            return integrate_linear_rounded_bins_cpu(cc_data->data(), cc_data->mask(), drifts, options);
        };
        cc_data.set_integrated_drift_plane(integrated_dedrift);
#if BLISS_CUDA
    } else if (compute_device.device_type == kDLCUDA) {
        auto integrated_dedrift = [cc_data = cc_copy, drifts, options]() {
            return integrate_linear_rounded_bins_cuda(cc_data->data(), cc_data->mask(), drifts, options);
        };
        cc_data.set_integrated_drift_plane(integrated_dedrift);
#endif
    } else {
        auto integrated_dedrift = [cc_data = cc_copy, drifts, options]() {
            return integrate_linear_rounded_bins_bland(cc_data->data(), cc_data->mask(), drifts, options);
        };
        cc_data.set_integrated_drift_plane(integrated_dedrift);
    }

    return cc_data;
}

scan bliss::integrate_drifts(scan scan_data, integrate_drifts_options options) {
    auto number_coarse_channels = scan_data.get_number_coarse_channels();
    for (auto cc_index = 0; cc_index < number_coarse_channels; ++cc_index) {
        auto cc = scan_data.read_coarse_channel(cc_index);
        *cc = integrate_drifts(*cc, options);
    }
    return scan_data;
}

observation_target bliss::integrate_drifts(observation_target target, integrate_drifts_options options) {
    for (auto &target_scan : target._scans) {
        target_scan = integrate_drifts(target_scan, options);
    }
    return target;
}

cadence bliss::integrate_drifts(cadence observation, integrate_drifts_options options) {
    for (auto &target : observation._observations) {
        target = integrate_drifts(target, options);
    }
    return observation;
}
