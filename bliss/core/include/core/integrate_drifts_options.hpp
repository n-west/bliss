#pragma once

#include "flag_values.hpp"

#include <bland/ndarray.hpp>

#include <cstdint>

namespace bliss {

struct integrate_drifts_options {
    bool desmear = true;

    float low_rate_Hz_per_sec = -5; // Hz/sec
    float high_rate_Hz_per_sec = 5; // Hz/sec
    // The search increment in numbers of drift resolutions (foff / total scan time)
    int resolution = 1;
    int round_to_multiple_of_data = true;
};


/**
 * a container to track how much rfi of each kind was involved in each drift integration
 */
struct integrated_flags {
    bland::ndarray low_spectral_kurtosis;
    bland::ndarray high_spectral_kurtosis;
    bland::ndarray sigma_clip;
    bland::ndarray::dev _device = default_device;

    integrated_flags(int64_t drifts, int64_t channels, bland::ndarray::dev device = default_device) :
            low_spectral_kurtosis(bland::ndarray({drifts, channels}, 0, bland::ndarray::datatype::uint8, device)),
            high_spectral_kurtosis(bland::ndarray({drifts, channels}, 0, bland::ndarray::datatype::uint8, device)),
            sigma_clip(bland::ndarray({drifts, channels}, 0, bland::ndarray::datatype::uint8, device))
            {}

    void set_device(bland::ndarray::dev device) {
        _device = device;
    }

    void set_device(std::string_view device) {
        bland::ndarray::dev dev = device;
        set_device(dev);
    }

    void push_device() {
        low_spectral_kurtosis = low_spectral_kurtosis.to(_device);
        high_spectral_kurtosis = high_spectral_kurtosis.to(_device);
        sigma_clip = sigma_clip.to(_device);
    }

};

} // namespace bliss
