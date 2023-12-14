#pragma once

#include <core/filterbank_data.hpp>

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
filterbank_data flag_filter_rolloff(filterbank_data fb_data, float rolloff_width);

} // namespace bliss
