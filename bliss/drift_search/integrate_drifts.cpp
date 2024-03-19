
#include "drift_search/integrate_drifts.hpp"
#include "kernels/drift_integration_bland.hpp"
#include "kernels/drift_integration_cpu.hpp"

#include <core/flag_values.hpp>

#include <bland/bland.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <thread>

using namespace bliss;

bland::ndarray bliss::integrate_drifts(const bland::ndarray &spectrum_grid, integrate_drifts_options options) {
    auto compute_device = spectrum_grid.device();

    if (compute_device.device_type == kDLCPU) {
        auto drift_grid = integrate_linear_rounded_bins_cpu(spectrum_grid, options);
        return drift_grid;
    } else if (compute_device.device_type == kDLCUDA) {
        auto drift_grid = integrate_linear_rounded_bins_bland(spectrum_grid, options);
        return drift_grid;
    } else {
        throw std::runtime_error("integrate_drifts not supported on this device");
    }
}

coarse_channel bliss::integrate_drifts(coarse_channel cc_data, integrate_drifts_options options) {
    auto compute_device = cc_data.device();

    auto cc_copy = std::make_shared<coarse_channel>(cc_data);
    if (compute_device.device_type == kDLCPU) {
        auto integrated_dedrift = [cc_data = cc_copy, options](){return integrate_linear_rounded_bins_cpu(cc_data->data(), cc_data->mask(), options);};
        cc_data.set_integrated_drift_plane(integrated_dedrift);
    } else if (compute_device.device_type == kDLCUDA) {
        auto integrated_dedrift = [cc_data = cc_copy, options](){return integrate_linear_rounded_bins_bland(cc_data->data(), cc_data->mask(), options);};
        cc_data.set_integrated_drift_plane(integrated_dedrift);
    }

    // cc_data.set_integrated_drift_plane(integrated_dedrift);
    return cc_data;
}

scan bliss::integrate_drifts(scan scan_data, integrate_drifts_options options) {
    auto number_coarse_channels = scan_data.get_number_coarse_channels();
    for (auto cc_index = 0; cc_index < number_coarse_channels; ++cc_index) {
        auto cc = scan_data.get_coarse_channel(cc_index);
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
