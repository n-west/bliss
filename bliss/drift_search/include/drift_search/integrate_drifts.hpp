
#pragma once

#include <bland/ndarray.hpp>
#include <core/doppler_spectrum.hpp>
#include <core/filterbank_data.hpp>

namespace bliss {

/**
 * Methods to select bins along a linear track in time-frequency spectrum estimate.
 *
 * Spectrum estimates are laid out in a 2d array of shape [time, frequency].
 *
 * LINEAR_ROUND: use m=time rows / freq cols. Select freq column = round((1/m)*time step). For example, if the
 * spectrum estimate contains 8 time rows, we compute 8 doppler trajectories per frequency with slopes through
 * spectrum computed as
 * 0 (freq bins) / 8 (slow time spectra) = 0
 * 1 (freq bins) / 8 (slow time spectra) = 0.125
 * 2 (freq bins) / 8 (slow time spectra) = 0.25
 * 3 (freq bins) / 8 (slow time spectra) = 0.375
 * 4 (freq bins) / 8 (slow time spectra) = 0.5
 * 5 (freq bins) / 8 (slow time spectra) = 0.625
 * 6 (freq bins) / 8 (slow time spectra) = 0.75
 * 7 (freq bins) / 8 (slow time spectra) = 0.875
 *
 * While following a track to sum components, the frequency column for a step is freq index = round(m*step) so that
 * doppler drift 0: sum(spectrum[0, 0], spectrum[1, 0], spectrum[2, 0], spectrum[3, 0],
 *                      spectrum[4, 1], spectrum[5, 1], spectrum[6, 1], spectrum[7, 1]).
 * Special note: for values of 0.5 (such as for doppler track spanning 4columns) this will round 0.5 up, which is
 * not always the default in some languages and libraries such as numpy and python which will round to the nearest
 * even which would give a slightly different track through spectrum.
 *
 * TAYLOR_TREE: use a tree-based method equivalent to turbo_seti, seticore, and first published by
 * Taylor, J. H, "A Sensitive Method for Detecting Dispersed Radio Emission." 1974 Astron. Astrophys. Suppl.
 *
 * HOUSTIN: not implemented yet, but will follow Ken Houston's rules for rounding
 */
// enum class spectrum_sum_method {
//     LINEAR_ROUND,
//     TAYLOR_TREE,
//     HOUSTON,
// };

/**
 * Integrate energy through a track in the spectrum according to the selected method for selecting tracks
 */
[[nodiscard]] bland::ndarray integrate_drifts(const bland::ndarray    &spectrum_grid,
                                              integrate_drifts_options options = integrate_drifts_options{
                                                      .desmear = true});

[[nodiscard]] doppler_spectrum integrate_drifts(filterbank_data          fil_data,
                                                integrate_drifts_options options = integrate_drifts_options{
                                                        .desmear = true});

// [[nodiscard]] doppler_spectrum integrate_drifts(observation_target       observations,
//                                                 integrate_drifts_options options = integrate_drifts_options{
//                                                         .desmear = true});

// [[nodiscard]] doppler_spectrum
// integrate_drifts(cadence observations, integrate_drifts_options options = integrate_drifts_options{.desmear = true});

} // namespace bliss
