#include <drift_search/integrate_and_search.hpp>

#include "bland/ndarray.hpp"
#include <drift_search/connected_components.hpp>

#include <drift_search/protohit_search.hpp>
#include <drift_search/hit_search.hpp>
// #include <drift_search/local_maxima.hpp>

#include <fmt/core.h>
#include <fmt/format.h>
#include <cstdint>

using namespace bliss;


coarse_channel bliss::integrate_and_search(coarse_channel cc_data, integrate_drifts_options options) {
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

scan bliss::integrate_and_search(scan scan_data, integrate_drifts_options options) {
    scan_data.add_coarse_channel_transform([options](coarse_channel cc) { return integrate_and_search(cc, options); });
    return scan_data;
}

observation_target bliss::integrate_and_search(observation_target target, integrate_drifts_options options) {
    for (auto &target_scan : target._scans) {
        target_scan = integrate_and_search(target_scan, options);
    }
    return target;
}

cadence bliss::integrate_and_search(cadence observation, integrate_drifts_options options) {
    for (auto &target : observation._observations) {
        target = integrate_and_search(target, options);
    }
    return observation;
}