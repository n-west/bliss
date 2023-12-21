#pragma once

#include <core/doppler_spectrum.hpp>
#include <core/noise_power.hpp>

namespace bliss {

using nd_coords = std::vector<int64_t>;

struct component {
    std::vector<nd_coords> locations;
    float                  max_integration = std::numeric_limits<float>::lowest();
    nd_coords              index_max;
};

struct hit {
    int64_t start_freq_index;
    float start_freq_MHz;
    int64_t rate_index;
    float drift_rate_Hz_per_sec;
    float   snr;
    int64_t bandwidth;
    double binwidth;
};

/**
 * S/N > snr_threshold Given a noise floor estimate of nf, signal amplitude s,
 * noise amplitude n...
 * S = (s - nf)**2
 * N = (n)**2         our estimate has already taken in to account noise floor
 * (s-nf)/(n) > sqrt(snr_threshold)
 * s-nf > n * sqrt(snr_threshold)
 * s > nf + sqrt(N * snr_threshold)
 * Since the noise power was estimate before integration, it also decreases by sqrt of integration length
*/
float compute_signal_threshold(const noise_stats &noise_stats, int64_t integration_length, float snr_threshold);

/**
 * Return a binary mask (dtype uint8) when doppler spectrum values fall above
 * the given SNR threshold based on estimated noise stats.
 * 
 * 1 indicates a given drift, frequency is above the SNR threshold
 * 0 indicates given drift, frequency is below the SNR threshold
*/
bland::ndarray hard_threshold_drifts(const bland::ndarray &dedrifted_spectrum,
                                     const noise_stats    &noise_stats,
                                     int64_t               integration_length,
                                     float                 snr_threshold);

/**
 * High level wrapper around finding drifting signals above a noise floor
*/
std::vector<hit> hit_search(doppler_spectrum dedrifted_spectrum, noise_stats noise_stats, float snr_threshold);

} // namespace bliss