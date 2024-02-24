#pragma once

#include <core/scan.hpp>
#include <core/cadence.hpp>

namespace bliss {

/**
 * Flag the frequency edges of give filterbank data
 *
 * Assume the filter roll-off occupies rolloff_fraction % of the band edge
 *
 * Example: 20% filter rolloff masks the lower and upper frequency 20% of bandwidth (total 40%)
 * index:       0 1 2 3 4 5 6 7 8 9
 * input mask:  o o o o o o o o o o
 * output mask: x x o o o o o o x x
 */
coarse_channel flag_filter_rolloff(coarse_channel cc_data, float rolloff_width);

scan flag_filter_rolloff(scan fb_data, float rolloff_width);

observation_target flag_filter_rolloff(observation_target observations, float rolloff_width);

cadence flag_filter_rolloff(cadence observations, float rolloff_width);

} // namespace bliss
