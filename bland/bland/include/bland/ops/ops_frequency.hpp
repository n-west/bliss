#pragma once


namespace bland {

struct ndarray;
struct ndarray_slice;


/**
 * forward fft
*/
ndarray fft(ndarray x);

/**
 * returns the square(abs(fftshift(fft(x))))
 * 
 * This is the fft-shifted frequency response of an LTI system
 * 
 * Assumes 1d
*/
ndarray fft_shift_mag_square(ndarray x);



} // namespace bland