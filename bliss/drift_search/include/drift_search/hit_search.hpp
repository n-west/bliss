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