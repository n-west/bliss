#pragma once

#include <core/scan.hpp>
#include <core/cadence.hpp>

namespace bliss {


/**
 * return a mask with flags set for grid elements which have magnitude above the defined threshold
 * 
*/
bland::ndarray flag_magnitude(const bland::ndarray &data, float threshold);

/**
 * return a masked copy of fb_data where the coarse_channel.data() is above the given threshold
 * 
*/
coarse_channel flag_magnitude(coarse_channel fb_data, float threshold);
coarse_channel flag_magnitude(coarse_channel fb_data);

/**
 * return a masked copy of fb_data where the scan.data() is above the given threshold
 * 
*/
scan flag_magnitude(scan fb_data, float threshold);

/**
 * return a masked copy of fb_data where the scan.data() is above a threshold.
 * When no threshold is given, this will internally compute a mean & stddev, then use a threshold
 * of mean + 10 * stddev
 * 
*/
scan flag_magnitude(scan fb_data);

observation_target flag_magnitude(observation_target observations, float threshold);
observation_target flag_magnitude(observation_target observations);

cadence flag_magnitude(cadence observations, float threshold);
cadence flag_magnitude(cadence observations);

} // namespace bliss
