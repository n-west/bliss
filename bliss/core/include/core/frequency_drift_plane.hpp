#pragma once

#include <bland/ndarray.hpp>

#include <cstdint>
#include <vector>

namespace bliss {

struct frequency_drift_plane {
    struct drift_rate {
            int index_in_plane;
            double drift_rate_slope = 0.0F;
            double drift_rate_Hz_per_sec = 0.0F;
            int desmeared_bins=1; // number of bins per spectra used to desmear
    };

    frequency_drift_plane(bland::ndarray drift_plane, integrated_flags drift_rfi) : _integrated_drifts(drift_plane), _dedrifted_rfi(drift_rfi) {}
    frequency_drift_plane(bland::ndarray drift_plane, integrated_flags drift_rfi, int64_t integration_steps, std::vector<drift_rate> dri) : 
        _integrated_drifts(drift_plane), _dedrifted_rfi(drift_rfi), _integration_steps(integration_steps), _drift_rate_info(dri) {
    }
    

    // slow-time steps passed through for a complete integration, the total number
    // of bins contributing to this integration is demsear_bins * integration_steps
    int64_t _integration_steps;

    std::vector<drift_rate> _drift_rate_info; // info for each drift rate searched (consider changing to map with key being integer number of unit drifts)

    // The actual frequency drift plane
    bland::ndarray _integrated_drifts;

    integrated_flags _dedrifted_rfi;

};

} // namespace bliss
