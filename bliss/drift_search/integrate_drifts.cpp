
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

    auto drift_grid = integrate_linear_rounded_bins(spectrum_grid, options);

    return drift_grid;
}

coarse_channel bliss::integrate_drifts(coarse_channel cc_data, integrate_drifts_options options) {
    auto [drift_grid, drift_rfi] = integrate_linear_rounded_bins_cpu(cc_data.data(), cc_data.mask(), options);
    // auto [drift_grid, drift_rfi] = integrate_linear_rounded_bins(fil_data.data(), fil_data.mask(), options);
    cc_data.integration_length(cc_data.data().size(0)); // length is just the amount of time
    cc_data.doppler_flags(drift_rfi);
    cc_data.doppler_spectrum(drift_grid);
    cc_data.dedoppler_options(options);
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
    std::vector<std::thread> drift_threads;
    for (auto &target_scan : target._scans) {
        drift_threads.emplace_back(
                [&target_scan, &options]() { target_scan = integrate_drifts(target_scan, options); });
    }
    for (auto &t : drift_threads) {
        t.join();
    }
    return target;
}

cadence bliss::integrate_drifts(cadence observation, integrate_drifts_options options) {

    std::vector<std::thread> drift_threads;
    for (auto &target : observation._observations) {
        drift_threads.emplace_back([&target, &options]() { target = integrate_drifts(target, options); });
    }
    for (auto &t : drift_threads) {
        t.join();
    }
    return observation;
}
