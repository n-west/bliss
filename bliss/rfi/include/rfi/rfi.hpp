#pragma once

#include <bland/ndarray.hpp>

namespace bliss {

    /**
     * Flag (returns a uint8 mask with 1s of flagged entries)
    */
    bland::ndarray flag_hot_pixels(const bland::ndarray &spectrum_grid);

} // bliss

