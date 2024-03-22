#pragma once

#include <bland/ndarray.hpp>
#include <core/scan.hpp>

namespace bliss {

/**
 * Spectral kurtosis is the time-domain kurtosis (4th order standardized moment) of a frequency
 * channel. This accepts spectra (X = |FFT(x)|) which has already been squared, so the spectral
 * kurtsosis is estimated with the estimator
 *
 * SK = (M N d + 1)/(M-1) * (M S_2 / S_1^2 -1)
 *
 * d is 1
 * N is the number of spectrograms already averaged per spectra we receive
 * M is the number of spectra in this population to estimate kurtosis over (commonly 8, 16, or 32)
 *
 * derived in "The Generalized Spectral Kurtosis Estimator" by Nita, G. M and Gary, D. E.
 * available at https://arxiv.org/abs/1005.4371
 *
 * The non-averaged estimator (N=1) and background derivation can be found in
 * "Radio Frequency Interference Excision Using Spectral-Domain Statistics"
 *
 */
[[nodiscard]] bland::ndarray estimate_spectral_kurtosis(const bland::ndarray &spectrum_grid, int32_t N, int32_t M, float d = 1.0);

[[nodiscard]] bland::ndarray estimate_spectral_kurtosis(coarse_channel &cc_data);

// bland::ndarray estimate_spectral_kurtosis(scan &fil_data);

} // namespace bliss