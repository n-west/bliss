#pragma once

#include <core/coarse_channel.hpp>
#include <core/scan.hpp>
#include <core/cadence.hpp>

namespace bliss {

/**
 * return a mask with flags set for grid elements which have magnitude above the defined threshold
 * 
*/
bland::ndarray flag_sigmaclip(const bland::ndarray &data, int max_iter=5, float low=3.0f, float high=4.0f);

/**
 * return a masked copy of fb_data where the coarse_channel.data() is above the given threshold
 * 
*/
coarse_channel flag_sigmaclip(coarse_channel fb_data, int max_iter=5, float low=3.0f, float high=4.0f);

/**
 * return a masked copy of fb_data where the scan.data() is above the given threshold
 * 
*/
scan flag_sigmaclip(scan fb_data, int max_iter=5, float low=3.0f, float high=4.0f);

/**
 * return a masked copy of fb_data where the scan.data() is above the given threshold
 * 
*/
observation_target flag_sigmaclip(observation_target target, int max_iter=5, float low=3.0f, float high=4.0f);

/**
 * return a masked copy of fb_data where the scan.data() is above the given threshold
 * 
*/
cadence flag_sigmaclip(cadence cadence_data, int max_iter=5, float low=3.0f, float high=4.0f);


}