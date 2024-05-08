#pragma once

#include "integrate_drifts_options.hpp" // integrated_flags

#include <bland/ndarray_deferred.hpp>

#include <cstdint>
#include <vector>

namespace bliss {

struct protohit_drift_info {
    float integration_adjusted_noise;
};

class frequency_drift_plane {
    public:
    struct drift_rate {
        int index_in_plane;
        double drift_rate_slope = 0.0F;
        double drift_rate_Hz_per_sec = 0.0F;
        int drift_channels_span = 0;
        int desmeared_bins=1; // number of bins per spectra used to desmear
    };

    frequency_drift_plane(bland::ndarray_deferred drift_plane, integrated_flags drift_rfi);
    frequency_drift_plane(bland::ndarray_deferred drift_plane, integrated_flags drift_rfi, int64_t integration_steps, std::vector<drift_rate> dri);

    int64_t integration_steps();

    std::vector<drift_rate> drift_rate_info();

    bland::ndarray integrated_drift_plane();

    integrated_flags integrated_rfi();

    void set_device(bland::ndarray::dev dev);

    void set_device(std::string_view dev_str);

    void push_device();

    private:
    // slow-time steps passed through for a complete integration, the total number
    // of bins contributing to this integration is demsear_bins * integration_steps
    int64_t _integration_steps;

    // info for each drift rate searched (consider changing to map with key being integer number of unit drifts)
    std::vector<drift_rate> _drift_rate_info;

    // The actual frequency drift plane
    bland::ndarray_deferred _integrated_drifts;

    integrated_flags _dedrifted_rfi;

    bland::ndarray::dev _device = default_device;
};

} // namespace bliss
