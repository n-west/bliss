
#include "drift_search/integrate_drifts.hpp"
#include "kernels/drift_integration_bland.hpp"
#include "kernels/drift_integration_cpu.hpp"

#include <core/flag_values.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <bland/bland.hpp>

using namespace bliss;

// namespace detail {

bland::ndarray bliss::integrate_drifts(const bland::ndarray &spectrum_grid, integrate_drifts_options options) {

    auto drift_grid = integrate_linear_rounded_bins(spectrum_grid, options);

    return drift_grid;
}

scan bliss::integrate_drifts(scan fil_data, integrate_drifts_options options) {
    auto [drift_grid, drift_rfi] = integrate_linear_rounded_bins_cpu(fil_data.data(), fil_data.mask(), options);
    // auto [drift_grid, drift_rfi] = integrate_linear_rounded_bins(fil_data.data(), fil_data.mask(), options);
    fil_data.integration_length(fil_data.data().size(0)); // length is just the amount of time
    fil_data.doppler_flags(drift_rfi);
    fil_data.doppler_spectrum(drift_grid);
    fil_data.dedoppler_options(options);
    return fil_data;
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
