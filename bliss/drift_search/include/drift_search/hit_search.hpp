#pragma once

#include <core/cadence.hpp>
#include <core/flag_values.hpp>
#include <core/hit.hpp>
#include <core/noise_power.hpp>
#include <core/scan.hpp>

#include <list>

namespace bliss {

using nd_coords = std::vector<int64_t>;
using rfi       = std::map<flag_values, uint8_t>; // TODO: not so elegant, but OKish?

struct component {
    std::vector<nd_coords> locations;
    float                  max_integration = std::numeric_limits<float>::lowest();
    rfi                    rfi_counts;
    nd_coords              index_max;
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
   {-7,  7}, {-6,  7}, {-5,  7}, {-4,  7}, {-3,  7},{-2,  7},  {-1, 7},  {0, 7},  {1, 7}, {2,  7}, {3,  7}, {4,  7}, {5,  7}, {6,  7}, {7,  7},
   {-7,  6}, {-6,  6}, {-5,  6}, {-4,  6}, {-3,  6},{-2,  6},  {-1, 6},  {0, 6},  {1, 6}, {2,  6}, {3,  6}, {4,  6}, {5,  6}, {6,  6}, {7,  6},
   {-7,  5}, {-6,  5}, {-5,  5}, {-4,  5}, {-3,  5},{-2,  5},  {-1, 5},  {0, 5},  {1, 5}, {2,  5}, {3,  5}, {4,  5}, {5,  5}, {6,  5}, {7,  5},
   {-7,  4}, {-6,  4}, {-5,  4}, {-4,  4}, {-3,  4},{-2,  4},  {-1, 4},  {0, 4},  {1, 4}, {2,  4}, {3,  4}, {4,  4}, {5,  4}, {6,  4}, {7,  4},
   {-7,  3}, {-6,  3}, {-5,  3}, {-4,  3}, {-3,  3},{-2,  3},  {-1, 3},  {0, 3},  {1, 3}, {2,  3}, {3,  3}, {4,  3}, {5,  3}, {6,  3}, {7,  3},
   {-7,  2}, {-6,  2}, {-5,  2}, {-4,  2}, {-3,  2},{-2,  2},  {-1, 2},  {0, 2},  {1, 2}, {2,  2}, {3,  2}, {4,  2}, {5,  2}, {6,  2}, {7,  2},
   {-7,  1}, {-6,  1}, {-5,  1}, {-4,  1}, {-3,  1},{-2,  1},  {-1, 1},  {0, 1},  {1, 1}, {2,  1}, {3,  1}, {4,  1}, {5,  1}, {6,  1}, {7,  1},
   {-7,  0}, {-6,  0}, {-5,  0}, {-4,  0}, {-3,  0},{-2,  0},  {-1, 0},  /* X  */ {1, 0}, {2,  0}, {3,  0}, {4,  0}, {5,  0}, {6,  0}, {7,  0},
   {-7, -1}, {-6, -1}, {-5, -1}, {-4, -1}, {-3, -1},{-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1}, {3, -1}, {4, -1}, {5, -1}, {6, -1}, {7, -1},
   {-7, -2}, {-6, -2}, {-5, -2}, {-4, -2}, {-3, -2},{-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}, {3, -2}, {4, -2}, {5, -2}, {6, -2}, {7, -2},
   {-7, -3}, {-6, -3}, {-5, -3}, {-4, -3}, {-3, -3},{-2, -3}, {-1, -3}, {0, -3}, {1, -3}, {2, -3}, {3, -3}, {4, -3}, {5, -3}, {6, -3}, {7, -3},
   {-7, -4}, {-6, -4}, {-5, -4}, {-4, -4}, {-3, -4},{-2, -4}, {-1, -4}, {0, -4}, {1, -4}, {2, -4}, {3, -4}, {4, -4}, {5, -4}, {6, -4}, {7, -4},
   {-7, -5}, {-6, -5}, {-5, -5}, {-4, -5}, {-3, -5},{-2, -5}, {-1, -5}, {0, -5}, {1, -5}, {2, -5}, {3, -5}, {4, -5}, {5, -5}, {6, -5}, {7, -5},
   {-7, -6}, {-6, -6}, {-5, -6}, {-4, -6}, {-3, -6},{-2, -6}, {-1, -6}, {0, -6}, {1, -6}, {2, -6}, {3, -6}, {4, -6}, {5, -6}, {6, -6}, {7, -6},
   {-7, -7}, {-6, -7}, {-5, -7}, {-4, -7}, {-3, -7},{-2, -7}, {-1, -7}, {0, -7}, {1, -7}, {2, -7}, {3, -7}, {4, -7}, {5, -7}, {6, -7}, {7, -7},
            // clang-format on
    };
};

/**
 * High level wrapper around finding drifting signals above a noise floor
 *
 * The returned scan is a copy of the given scan with the hits field set
 */
std::list<hit> hit_search(coarse_channel dedrifted_scan, hit_search_options options = {});

/**
 * High level wrapper around finding drifting signals above a noise floor
 *
 * The returned scan is a copy of the given scan with the hits field set
 */
scan hit_search(scan dedrifted_scan, hit_search_options options = {});

/**
 * High-level hit search over scans within an observation target
 *
 * The returned observation_target is a copy of the given observation_target with hits set for all scans of the target
 */
observation_target hit_search(observation_target dedrifted_target, hit_search_options options = {});

/**
 * High-level hit search over an entire cadence
 *
 * The returned cadence is a copy of the given cadence with hits set inside each scan
 */
cadence hit_search(cadence dedrifted_cadence, hit_search_options options = {});

} // namespace bliss