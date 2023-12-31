#pragma once

#include <core/filterbank_data.hpp>
#include <core/cadence.hpp>

namespace bliss {


/**
 * return a mask with flags set for grid elements which have magnitude above the defined threshold
 * 
*/
bland::ndarray flag_magnitude(const bland::ndarray &data, float threshold);

/**
 * return a masked copy of fb_data where the filterbank_data.data() is above the given threshold
 * 
*/
filterbank_data flag_magnitude(filterbank_data fb_data, float threshold);

/**
 * return a masked copy of fb_data where the filterbank_data.data() is above a threshold.
 * When no threshold is given, this will internally compute a mean & stddev, then use a threshold
 * of mean + 10 * stddev
 * 
*/
filterbank_data flag_magnitude(filterbank_data fb_data);

observation_target flag_magnitude(observation_target observations, float threshold);
observation_target flag_magnitude(observation_target observations);

cadence flag_magnitude(cadence observations, float threshold);
cadence flag_magnitude(cadence observations);

} // namespace bliss
