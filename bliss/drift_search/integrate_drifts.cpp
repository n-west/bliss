
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
    auto maximum_drift_time_span = time_steps - 1;

    // Convert the drift options to specific drift info
    auto foff_Hz = (foff)*1e6;
    double unit_drift_resolution = foff_Hz/((time_steps-1) * tsamp);
    // Round the drift bounds to multiple of unit_drift_resolution
    auto search_resolution_Hz_sec = unit_drift_resolution * options.resolution;
    auto rounded_low_drift_Hz_sec = std::round(options.low_rate_Hz_per_sec / unit_drift_resolution) * unit_drift_resolution;
    auto rounded_high_drift_Hz_sec = std::round(options.high_rate_Hz_per_sec / unit_drift_resolution) * unit_drift_resolution;

    auto number_drifts = std::abs((rounded_high_drift_Hz_sec - rounded_low_drift_Hz_sec) / (search_resolution_Hz_sec));
    fmt::print("INFO: Searching {} drift rates from {} Hz/sec to {} Hz/sec in increments of {} Hz/sec\n", number_drifts, rounded_low_drift_Hz_sec, rounded_high_drift_Hz_sec, std::abs(search_resolution_Hz_sec));
    std::vector<frequency_drift_plane::drift_rate> drift_rate_info;
    drift_rate_info.reserve(number_drifts);
    for (int index = 0; index < number_drifts; ++index) {
        auto drift_rate = rounded_low_drift_Hz_sec + index * std::abs(search_resolution_Hz_sec);
        frequency_drift_plane::drift_rate rate;
        rate.index_in_plane = index;
        // You'll get automatic truncation without intentional rounding
        rate.drift_channels_span = std::lround(drift_rate * ((maximum_drift_time_span)*tsamp)/(foff_Hz));

        // The actual slope of that drift (number channels / number time steps)
        // Should be equivalent to a unitless drift rate (drift_rate * tsamp/foff)
        auto m = static_cast<float>(rate.drift_channels_span) / static_cast<float>(maximum_drift_time_span);

        rate.drift_rate_slope = m;
        rate.drift_rate_Hz_per_sec = drift_rate;
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


coarse_channel bliss::integrate_drifts(coarse_channel cc_data, integrate_drifts_options options) {
    auto compute_device = cc_data.device();

    auto drifts = compute_drifts(cc_data.ntsteps(), cc_data.foff(), cc_data.tsamp(), options);

    if (compute_device.device_type == kDLCPU) {
        auto integrated_dedrift = integrate_linear_rounded_bins_cpu(cc_data.data(), cc_data.mask(), drifts, options);
        cc_data.set_integrated_drift_plane(integrated_dedrift);
#if BLISS_CUDA
    } else if (compute_device.device_type == kDLCUDA) {
        auto integrated_dedrift = integrate_linear_rounded_bins_cuda(cc_data.data(), cc_data.mask(), drifts, options);
        cc_data.set_integrated_drift_plane(integrated_dedrift);
#endif
    } else {
        auto integrated_dedrift = integrate_linear_rounded_bins_bland(cc_data.data(), cc_data.mask(), drifts, options);
        cc_data.set_integrated_drift_plane(integrated_dedrift);
    }

    return cc_data;
}

scan bliss::integrate_drifts(scan scan_data, integrate_drifts_options options) {
    scan_data.add_coarse_channel_transform([options](coarse_channel cc) { return integrate_drifts(cc, options); });
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
