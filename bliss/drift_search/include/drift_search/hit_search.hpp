#pragma once

#include <core/doppler_spectrum.hpp>
#include <core/noise_power.hpp>

#include <flaggers/flag_values.hpp>

#include <map>

namespace bliss {

using nd_coords = std::vector<int64_t>;
using rfi = std::map<flag_values, uint8_t>; // TODO: not so elegant, but OKish?

struct component {
    std::vector<nd_coords>         locations;
    float                          max_integration = std::numeric_limits<float>::lowest();
    rfi rfi_counts;
    nd_coords                      index_max;
};

struct hit {
    int64_t start_freq_index;
    float   start_freq_MHz;
    int64_t rate_index;
    float   drift_rate_Hz_per_sec;
    float   snr;
    int64_t bandwidth;
    double  binwidth;
    rfi rfi_counts;
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

enum class hit_search_methods { CONNECTED_COMPONENTS, LOCAL_MAXIMA };

struct hit_search_options {
    hit_search_methods method = hit_search_methods::CONNECTED_COMPONENTS;

    /**
     * threshold (linear SNR) that integrated power must be above to be considered a hit
     */
    float snr_threshold = 10.0f;

    std::vector<nd_coords> neighborhood = {
            // clang-format off
            {-1, 1},  {1, 0},  {1, 1},
            {0, -1},  /* X  */ {0, 1},
            {-1, -1}, {-1, 0}, {1, -1},
            // {2, 0},
            // {0, 2},
            // {-2, 0},
            // {0, -2}
            // clang-format on
    };
};

/**
 * High level wrapper around finding drifting signals above a noise floor
 */
std::vector<hit>
hit_search(doppler_spectrum dedrifted_spectrum, noise_stats noise_stats, hit_search_options options = {});

} // namespace bliss