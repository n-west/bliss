
#pragma once

#include <core/filterbank_data.hpp>
#include <core/cadence.hpp>

namespace bliss {



/**
 * return a mask with flags set for grid elements which have non gaussian spectral kurtosis
 * 
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
*/
bland::ndarray flag_spectral_kurtosis(const bland::ndarray &data, int64_t N, int64_t M, float d, float lower_threshold, float upper_threshold);

/**
 * Flag the filterbank data based on spectral kurtosis bounds and return the flagged filterbank
*/
filterbank_data flag_spectral_kurtosis(filterbank_data fb_data, float lower_threshold=0.05f, float upper_threshold=0.05f);

/**
 * Flag all filterbanks in an observation target
*/
observation_target flag_spectral_kurtosis(observation_target observations, float lower_threshold, float upper_threshold);

/**
 * Flag all filterbanks in the cadence with given SK estimate
*/
cadence flag_spectral_kurtosis(cadence fb_data, float lower_threshold=0.05f, float upper_threshold=0.05f);

} // namespace bliss
