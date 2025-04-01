#pragma once

#include <cstdint>
#include <map>

namespace bliss {
/**
 * flag values are used to indicate further downstream what kind of flag some sample was flagged with
 */
enum flag_values {
    unflagged              = 0,
    filter_rolloff         = 1 << 0,
    low_spectral_kurtosis  = 1 << 1,
    high_spectral_kurtosis = 1 << 2,
    RESERVED_0             = 1 << 3,
    magnitude              = 1 << 4,
    sigma_clip             = 1 << 5,
    RESERVED_1             = 1 << 6,
    RESERVED_2             = 1 << 7,
};

using rfi = std::map<flag_values, uint8_t>; // TODO: not so elegant, but OKish?

} // namespace bliss
