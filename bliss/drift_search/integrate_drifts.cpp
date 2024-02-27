
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

    auto drift_grid = integrate_linear_rounded_bins_cpu(spectrum_grid, options);

    return drift_grid;
}

coarse_channel bliss::integrate_drifts(coarse_channel cc_data, integrate_drifts_options options) {
    auto integrated_dedrift = integrate_linear_rounded_bins_cpu(cc_data.data(), cc_data.mask(), options);

    cc_data.set_integrated_drift_plane(integrated_dedrift);
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
