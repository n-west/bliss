#pragma once

#include <core/filterbank_data.hpp>

namespace bliss {


/**
 * return a mask with flags set for grid elements which have very large magnitudes
 * 
*/
filterbank_data flag_magnitude(filterbank_data fb_data, float sigma);

} // namespace bliss
