#pragma once

#include "hit_search_options.hpp"

#include <core/protohit.hpp> // using rfi = map<>
#include <core/cadence.hpp>
#include <core/flag_values.hpp>
#include <core/hit.hpp>
#include <core/noise_power.hpp>
#include <core/scan.hpp>

#include <bland/stride_helper.hpp>

#include <list>

namespace bliss {

using rfi       = std::map<flag_values, uint8_t>; // TODO: not so elegant, but OKish?

// struct protohit {
//     std::vector<bland::nd_coords> locations;
//     float                  max_integration = std::numeric_limits<float>::lowest();
//     float                  desmeared_noise;
//     rfi                    rfi_counts;
//     bland::nd_coords       index_max;
// };

/**
  * When the signal amplitude is snr_threshold above the noise floor, we have a 'prehit' (a signal that naively
  * passes a hard threshold) that is when S/N > snr_threshold Given a noise floor estimate of nf, signal power above threshold S,
  * noise power N...
  * (S - noise_floor) / N > snr_threshold
  * S - noise_floor > N * snr_threshold
  * S > noise_floor + N * snr_threshold
  * We have incoherently integrated (with mean) l bins, so adjust the noise power by sqrt(l)
 */
float compute_signal_threshold(noise_stats &noise_stats, int64_t integration_length, float snr_threshold);

std::vector<std::pair<float, float>> compute_noise_and_snr_thresholds(noise_stats   &noise_stats,
                                            int64_t                                        integration_length,
                                            std::vector<frequency_drift_plane::drift_rate> drift_rates,
                                            float                                          snr_threshold);

/**
 * Return a binary mask (dtype uint8) when doppler spectrum values fall above
 * the given SNR threshold based on estimated noise stats.
 *
 * 1 indicates a given drift, frequency is above the SNR threshold
 * 0 indicates given drift, frequency is below the SNR threshold
 */
bland::ndarray hard_threshold_drifts(const bland::ndarray &dedrifted_spectrum,
                                     noise_stats    &noise_stats,
                                     int64_t               integration_length,
                                     float                 snr_threshold);



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