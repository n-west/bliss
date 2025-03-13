#pragma once

#include <bland/ndarray.hpp>
#include <core/cadence.hpp>
#include <core/scan.hpp>
#include <core/coarse_channel.hpp>

#include "hit_search_options.hpp"
#include "integrate_drifts_options.hpp"

// #include <core/flag_values.hpp>
// #include <core/hit.hpp>
// #include <core/noise_power.hpp>

// #include <bland/stride_helper.hpp>

#include <list>

namespace bliss {

// /**
//  * Integrate energy through a track in the spectrum according to the selected method for selecting tracks
//  */
// [[nodiscard]] coarse_channel
// integrate_drifts(coarse_channel cc_data, integrate_drifts_options options = integrate_drifts_options{.desmear = true});

// [[nodiscard]] scan integrate_drifts(scan                     scan_data,
//                                     integrate_drifts_options options = integrate_drifts_options{.desmear = true});

// /**
//  * Integrate energy through linear tracks in the scans of given observation target
//  *
//  * The returned observation_target is a copy of the given observation_target with valid dedrifted_coarse_channel
//  * fields of each scan.
//  */
// [[nodiscard]] observation_target integrate_drifts(observation_target       target,
//                                                   integrate_drifts_options options = integrate_drifts_options{
//                                                           .desmear = true});

// /**
//  * Integrate energy through linear tracks in the scans of the given cadence
//  *
//  * The returned cadence is a copy of the given cadence with valid dedrifted_coarse_channel for each scan
//  * in each observation target.
//  */
// [[nodiscard]] cadence integrate_drifts(cadence                  observations,
//                                        integrate_drifts_options options = integrate_drifts_options{.desmear = true});



// /**
//  * High level wrapper around finding drifting signals above a noise floor
//  *
//  * The returned scan is a copy of the given scan with the hits field set
//  */
// std::list<hit> hit_search(coarse_channel dedrifted_scan, hit_search_options options = {});

// /**
//  * High level wrapper around finding drifting signals above a noise floor
//  *
//  * The returned scan is a copy of the given scan with the hits field set
//  */
// scan hit_search(scan dedrifted_scan, hit_search_options options = {});

// /**
//  * High-level hit search over scans within an observation target
//  *
//  * The returned observation_target is a copy of the given observation_target with hits set for all scans of the target
//  */
// observation_target hit_search(observation_target dedrifted_target, hit_search_options options = {});

// /**
//  * High-level hit search over an entire cadence
//  *
//  * The returned cadence is a copy of the given cadence with hits set inside each scan
//  */
// cadence hit_search(cadence dedrifted_cadence, hit_search_options options = {});

coarse_channel integrate_and_search(coarse_channel cc_data, integrate_drifts_options dedrift_options, hit_search_options hit_search_options);

scan integrate_and_search(scan scan_data, integrate_drifts_options dedrift_options, hit_search_options hit_search_options);

observation_target integrate_and_search(observation_target observations, integrate_drifts_options dedrift_options, hit_search_options hit_search_options);

cadence integrate_and_search(cadence observations, integrate_drifts_options dedrift_options, hit_search_options hit_search_options);

}