
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
