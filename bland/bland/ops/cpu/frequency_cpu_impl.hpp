#pragma once

#include <bland/ndarray.hpp>


namespace bland {
namespace cpu {

ndarray fft(ndarray a, ndarray &out);

ndarray fft_shift_mag_square(ndarray a, ndarray &out);

} // namespace cpu
} // namespace bland
