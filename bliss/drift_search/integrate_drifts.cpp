
#include "drift_search/integrate_drifts.hpp"

#include "drift_search/compute_drift_rates.hpp"

#include "kernels/drift_integration_bland.hpp"
#include "kernels/drift_integration_cpu.hpp"
#if BLISS_CUDA
#include "kernels/drift_integration_cuda.cuh"
#endif

#include <bland/bland.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

using namespace bliss;


frequency_drift_plane bliss::integrate_drifts(bland::ndarray data, bland::ndarray mask, std::vector<frequency_drift_plane::drift_rate> drifts, integrate_drifts_options options) {
    auto compute_device = data.device();
    // TODO: check that mask device matches, error out as appropriate

    if (compute_device.device_type == kDLCPU) {
        return integrate_linear_rounded_bins_cpu(data, mask, drifts, options);
#if BLISS_CUDA
    } else if (compute_device.device_type == kDLCUDA) {
        return integrate_linear_rounded_bins_cuda(data, mask, drifts, options);
#endif
    } else {
        return integrate_linear_rounded_bins_bland(data, mask, drifts, options);
    }
}

coarse_channel bliss::integrate_drifts(coarse_channel cc_data, integrate_drifts_options options) {
    auto drifts = compute_drifts(cc_data.ntsteps(), cc_data.foff(), cc_data.tsamp(), options);

    auto integrated_dedrift = integrate_drifts(cc_data.data(), cc_data.mask(), drifts, options);
    cc_data.set_integrated_drift_plane(integrated_dedrift);

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
